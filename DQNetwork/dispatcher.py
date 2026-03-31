# dispatcher.py
"""
Heuristic dispatcher for shared-ride DRT using SUMO/TraCI.

Design:
  - Runs in 1-second simulation steps, grouped into 10-step ticks.
  - At each tick end, checks for meaningful events:
      * new request arrived
      * passenger picked up
      * passenger dropped off
  - If a meaningful event occurred AND pending requests exist:
      * generates all feasible CandidateInsertions
      * scores each candidate with a heuristic scoring function
      * dispatches the highest-scoring candidate
      * prints top-5 candidates
  - Prints a tick summary whenever a meaningful event fires.

Scoring (lower raw cost → higher score, score = -cost):
  - detour penalty  : total added delay imposed on all existing onboard passengers
  - waiting penalty : current waiting time of the new passenger being inserted
  - activation penalty : large flat cost if the chosen taxi was fully IDLE
                         (discourages spinning up extra vehicles)

Usage:
    python dispatcher.py --cfg path/to/map.sumocfg [--gui] [--step-length 1.0]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from typing import Callable, Optional
import csv
from pathlib import Path
import math
import traci

# ---------------------------------------------------------------------------
# Score normalization settings
# ---------------------------------------------------------------------------

NORMALIZE_SCORE_COMPONENTS = True
MAX_SERVICE_SEQ = 20

# For very large / skewed terms, compress first before normalization
LOG_SCALE_KEYS = {
    "new_wait_viol",
    "new_ride_viol",
    "exist_wait_viol",
    "exist_ride_viol",
    "workload",
    "imbalance",
}

# Optional hard clipping after normalization to stop extreme outliers
NORMALIZED_CLIP_VALUE = 3.0

from DRTDataclass import (
    CandidateInsertion,
    Request,
    RequestStatus,
    Stop,
    StopType,
    TaxiPlan,
    TaxiStatus,
    TickContext,
    TickOutcome,
    IntervalAccumulator,
)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(log_file: str) -> logging.Logger:
    """
    Returns a logger that writes to both a .log file and the console.
    The file gets everything; the console only shows INFO and above.

    Log file is named with a timestamp so each run produces a fresh file:
        dispatcher_20250312_143022.log
    """
    logger = logging.getLogger("drt_dispatcher")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")   # plain text, no level prefix clutter

    # --- file handler (all output) ---
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- console handler (mirror to terminal as well) ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# module-level logger placeholder; replaced in main()
log: logging.Logger = logging.getLogger("drt_dispatcher")


def _log(msg: str) -> None:
    """Drop-in replacement for print() — writes to log file and console."""
    log.info(msg)


W_MAX_ONBOARD_DELAY = 1.55   # protect the worst existing passenger from large delay
W_AVG_ONBOARD_DELAY = 0.18   # mild average-delay smoothing for existing riders
W_WAIT_SO_FAR      = 0.18    # request already waited before this decision
W_FUTURE_WAIT      = 0.3    # strong penalty for extra time until pickup from now
W_TOTAL_WAIT       = 0.10    # light extra guard on total wait since request time
W_ROUTE            = 0.2    # mild penalty for route extension
W_ACTIVATION       = 0.25   # lighter cost for waking an idle taxi
W_WORKLOAD         = 0.5  # penalise long remaining suffix on one taxi
W_IMBALANCE        = 0.2   # penalise overloading one taxi relative to the other
W_SHARE_BONUS      = 0.1     # reward compact, useful pooling
W_NEW_WAIT_VIOL = 0.25
W_NEW_RIDE_VIOL = 0.3
W_EXIST_WAIT_VIOL = 0.5
W_EXIST_RIDE_VIOL = 0.25
# FOR REQUEST
REQ_BASE_WAIT = 120.0      # everyone gets 2 minutes baseline
REQ_ALPHA = 0.5            # add 0.5 sec tolerated wait per 1 sec direct trip
REQ_MAX_WAIT_CAP = 600.0   # cap at 10 minutes

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _route_time(from_edge: str, to_edge: str, vtype: str) -> float:
    """Return estimated travel time between two edges. Returns inf if unreachable."""
    if from_edge == to_edge:
        return 0.0
    try:
        route = traci.simulation.findRoute(from_edge, to_edge, vtype, routingMode=1)
        return route.travelTime if route.travelTime > 0 else float("inf")
    except Exception:
        return float("inf")


def _estimate_eta_chain(start_edge: str, stops: list[Stop], vtype: str, now: float) -> list[float]:
    """
    Walk the stop chain from start_edge and return an ETA for each stop.
    ETAs are absolute simulation times.
    """
    etas: list[float] = []
    current_edge = start_edge
    current_time = now
    for stop in stops:
        tt = _route_time(current_edge, stop.edge_id, vtype)
        current_time += tt
        etas.append(current_time)
        current_edge = stop.edge_id
    return etas


def _build_request_lookup_by_res_id(requests: dict[str, Request]) -> dict[str, Request]:
    """Build reservation_id -> Request lookup from the person-keyed request dict."""
    return {req.request_id: req for req in requests.values()}


def _clone_stops(stops: list[Stop]) -> list[Stop]:
    """Deep-copy stop objects so rollback restores the exact pre-tick plan."""
    return [copy.deepcopy(s) for s in stops]


def _serialize_dispatch_res_ids(plan: TaxiPlan) -> list[str]:
    """
    Convert the full remaining stop sequence into the reservation-id sequence
    expected by traci.vehicle.dispatchTaxi.

    Important: SUMO expects the COMPLETE future reservation chain, not just
    upcoming pickups. Therefore we emit one reservation id per remaining stop:
      - pickup  -> reservation id once
      - dropoff -> same reservation id again

    Examples
    --------
    stops = [PU(21), DO(21), PU(22), DO(22)] -> ['21', '21', '22', '22']
    stops = [DO(20), PU(21), DO(21)]         -> ['20', '21', '21']
    """
    return [s.request_id for s in plan.stops]


def _eligible_taxis_for_tick(
    taxi_plans: dict[str, TaxiPlan],
    requests: dict[str, Request],
    new_pickups: list[str],
    new_dropoffs: list[str],
) -> set[str]:
    """
    Taxis that may be reconsidered this tick.

    Broader policy than the earlier stop-boundary-only version:
      - allow every taxi still present in the local fleet state
      - preserve SUMO stability by freezing the first already-committed stop
        during candidate generation for taxis that already have a plan

    This keeps periodic dispatch, but makes ride-sharing materially more dynamic
    because moving taxis can accept new requests without waiting for the exact
    tick when they hit a pickup/dropoff boundary.
    """
    return set(taxi_plans.keys())


def enumerate_all_raw_candidates(
    request: Request,
    taxi_plans: dict[str, TaxiPlan],
    now: float,
    eligible_taxi_ids: set[str] | None = None,
) -> list[CandidateInsertion]:
    """
    Debug-only enumerator:
    returns ALL insertion combinations that satisfy only:
      - taxi is eligible
      - do_idx > pu_idx
    No capacity / max-wait / ride-time / delay constraints are applied.
    """
    if eligible_taxi_ids is None:
        eligible_taxi_ids = set(taxi_plans.keys())

    raw_candidates: list[CandidateInsertion] = []

    active_vehicle_ids = set(traci.vehicle.getIDList())

    for taxi_id, plan in taxi_plans.items():
        if taxi_id not in eligible_taxi_ids:
            continue
        if taxi_id not in active_vehicle_ids:
            continue

        try:
            vtype = traci.vehicle.getTypeID(taxi_id)
        except traci.TraCIException:
            continue

        n = len(plan.stops)

        # use the same freeze rule as your real generator
        if n == 0:
            frozen_prefix = 0
        elif plan.stops[0].stop_type == StopType.DROPOFF:
            frozen_prefix = 0
        else:
            frozen_prefix = 1

        for pu_idx in range(frozen_prefix, n + 1):
            for do_idx in range(pu_idx + 1, n + 2):
                pu_stop = Stop(
                    StopType.PICKUP,
                    request.request_id,
                    request.person_id,
                    request.from_edge,
                )
                do_stop = Stop(
                    StopType.DROPOFF,
                    request.request_id,
                    request.person_id,
                    request.to_edge,
                )

                new_stops = (
                    plan.stops[:pu_idx]
                    + [pu_stop]
                    + plan.stops[pu_idx:do_idx - 1]
                    + [do_stop]
                    + plan.stops[do_idx - 1:]
                )

                etas = _estimate_eta_chain(plan.current_edge, new_stops, vtype, now)

                pu_eta = etas[new_stops.index(pu_stop)] if etas else 0.0
                do_eta = etas[new_stops.index(do_stop)] if etas else 0.0
                added_route_time = max(0.0, etas[-1] - now) if etas else 0.0

                for stop_obj, eta_val in zip(new_stops, etas):
                    stop_obj.eta = eta_val

                raw_candidates.append(
                    CandidateInsertion(
                        request_id=request.request_id,
                        taxi_id=taxi_id,
                        pickup_index=pu_idx,
                        dropoff_index=do_idx,
                        resulting_stops=new_stops,
                        added_route_time=added_route_time,
                        pickup_eta_new=pu_eta,
                        dropoff_eta_new=do_eta,
                        max_existing_delay=0.0,
                        avg_existing_delay=0.0,
                        is_feasible=True,   # debug only; means "enumerated", not "validated"
                    )
                )

    return raw_candidates


def _print_all_raw_candidates(
    candidates: list[CandidateInsertion],
    request: Request,
    now: float,
) -> None:
    _log(
        f"\n  ┌─ ALL RAW CANDIDATES for request {request.request_id}"
        f" (person {request.person_id}) waited {request.waiting_time(now):.1f}s ─┐"
    )

    if not candidates:
        _log("  │  (none)")
        _log("  └" + "─" * 70)
        return

    for rank, c in enumerate(candidates, 1):
        route_preview = " -> ".join(
            f"{'PU' if s.stop_type == StopType.PICKUP else 'DO'}({s.request_id})"
            for s in c.resulting_stops
        )
        _log(
            f"  │  #{rank:<2} taxi={c.taxi_id:>6}  "
            f"pu_idx={c.pickup_index}  do_idx={c.dropoff_index}  "
            f"pu_eta={c.pickup_eta_new:7.1f}s  "
            f"route={route_preview}"
        )

    _log("  └" + "─" * 70)


def generate_candidates(
    request: Request,
    taxi_plans: dict[str, TaxiPlan],
    requests: dict[str, Request],
    now: float,
    eligible_taxi_ids: set[str] | None = None,
    request_lookup_by_res_id: dict[str, Request] | None = None,
    max_ride_factor: float = 1.5,
    max_wait: float = 300.0,
) -> list[CandidateInsertion]:
    """
    For the given request, enumerate every feasible (taxi, pickup_idx, dropoff_idx)
    insertion across all taxis, plus a DEFER pseudo-action.

    Under the periodic-dispatch design, we reconsider all active taxis.
    To avoid destabilising SUMO, taxis with an existing suffix plan keep their
    very first already-committed stop frozen; new requests may only be inserted
    after that frozen prefix. This gives much more ride-sharing flexibility than
    stop-boundary-only replanning while still avoiding hard rewrites of the
    immediate next stop currently being executed by the taxi.
    """
    if eligible_taxi_ids is None:
        eligible_taxi_ids = set(taxi_plans.keys())
    if request_lookup_by_res_id is None:
        request_lookup_by_res_id = _build_request_lookup_by_res_id(requests)

    candidates: list[CandidateInsertion] = []

    active_vehicle_ids = set(traci.vehicle.getIDList())
    for taxi_id, plan in taxi_plans.items():
        if taxi_id not in eligible_taxi_ids:
            continue
        # skip taxis that have left the simulation (teleported/finished)
        if taxi_id not in active_vehicle_ids:
            continue
        try:
            vtype = traci.vehicle.getTypeID(taxi_id)
        except traci.TraCIException:
            continue

        n = len(plan.stops)
        req_max_wait = getattr(request, "max_wait", max_wait)
        req_max_ride_factor = getattr(request, "max_ride_factor", max_ride_factor)

        # Freeze the very first committed stop for taxis that already have a
        # route. New insertions may only affect the suffix after that point.
        # More permissive freeze rule:
        # - no stops          -> free insertion anywhere
        # - first stop DROP   -> allow inserting before it
        # - first stop PICKUP -> keep the first stop frozen for stability
        # if n == 0:
        #     frozen_prefix = 0
        # elif plan.stops[0].stop_type == StopType.DROPOFF:
        #     frozen_prefix = 1
        # else:
        #     frozen_prefix = 1
        if n >= MAX_SERVICE_SEQ:
            continue
        if n == 0:
            frozen_prefix = 0
        else:
            first_pickup_idx = next(
                (i for i, s in enumerate(plan.stops) if s.stop_type == StopType.PICKUP),
                None
            )

            if first_pickup_idx is None:
                # only dropoffs remain -> no future pickup to protect
                frozen_prefix = 0
            else:
                # preserve everything up to and including the first planned pickup
                frozen_prefix = first_pickup_idx + 1

        # Baseline concurrent count = passengers already onboard with no
        # PICKUP stop remaining (they are in the taxi right now).
        onboard_req_ids_in_stops = {
            s.request_id for s in plan.stops if s.stop_type == StopType.PICKUP
        }
        # passenger inside the taxi now
        baseline_onboard = sum(
            1 for rid in plan.onboard_request_ids
            if rid not in onboard_req_ids_in_stops
        )

        # Pre-compute original ETAs once per taxi (reused for delay calc)
        orig_etas = _estimate_eta_chain(plan.current_edge, plan.stops, vtype, now)
        orig_eta_map = {id(s): orig_etas[i] for i, s in enumerate(plan.stops)}

        for pu_idx in range(frozen_prefix, n + 1):
            for do_idx in range(pu_idx + 1, n + 2):
                # Build new stop list with fresh Stop objects so ETAs can be
                # written back without aliasing the original plan.stops entries.
                pu_stop = Stop(StopType.PICKUP,  request.request_id,
                               request.person_id, request.from_edge)
                do_stop = Stop(StopType.DROPOFF, request.request_id,
                               request.person_id, request.to_edge)

                new_stops = (plan.stops[:pu_idx] + [pu_stop] +
                             plan.stops[pu_idx:do_idx - 1] + [do_stop] +
                             plan.stops[do_idx - 1:])

                etas = _estimate_eta_chain(plan.current_edge, new_stops, vtype, now)

                # ── 1. Capacity check ──────────────────────────────────────
                # Walk the stop list counting pickups/dropoffs starting from
                # baseline_onboard (passengers already in vehicle).
                concurrent = baseline_onboard
                peak = concurrent
                for s in new_stops:
                    concurrent += 1 if s.stop_type == StopType.PICKUP else -1
                    peak = max(peak, concurrent)
                if peak > plan.capacity:
                    # _log(f"TRIGGER PEAK")
                    continue

                # # ── 2. New passenger pickup wait ───────────────────────────
                pu_idx_in_new = new_stops.index(pu_stop)
                pu_eta = etas[pu_idx_in_new]
                # if pu_eta - request.request_time > req_max_wait:
                #     # _log(f"TRIGGER PICKUP WAIT")
                #     continue

                # # ── 3. New passenger ride time ─────────────────────────────
                do_idx_in_new = new_stops.index(do_stop)
                do_eta = etas[do_idx_in_new]
                # actual_ride = do_eta - pu_eta
                # if request.direct_travel_time > 0:
                #     if actual_ride > req_max_ride_factor * request.direct_travel_time:
                #         _log(f"Candidate rejected NEW REQ {request.request_id} on taxi {taxi_id}: \n ride time {actual_ride:.1f}s > {req_max_ride_factor} * {request.direct_travel_time:.1f} \n PU idx={pu_idx} DO idx={do_idx}\nTAXI STOPS: {new_stops}")
                #         continue
                new_wait = max(0.0, pu_eta - request.request_time)
                allowed_new_wait = req_max_wait
                new_wait_violation = max(0.0, new_wait - allowed_new_wait)

                actual_ride = do_eta - pu_eta
                allowed_new_ride = (
                    req_max_ride_factor * request.direct_travel_time
                    if request.direct_travel_time > 0 else float("inf")
                )
                new_ride_violation = max(0.0, actual_ride - allowed_new_ride)

                # ── 4. Existing passenger constraints ──────────────────────
                # feasible = True
                max_existing_delay = 0.0
                total_existing_delay = 0.0
                max_pickup_delay = 0.0
                existing_count = 0

                existing_wait_violation_sum = 0.0
                existing_wait_violation_max = 0.0
                existing_ride_violation_sum = 0.0
                existing_ride_violation_max = 0.0


                # the following is used to check the feasibility so that:
                # 1) existing not-yet-picked-up passengers do not exceed max wait
                # 2) existing passengers do not exceed max ride factor
                # 3) delay metrics for existing stops are computed
                for i, s in enumerate(new_stops):
                    # skip the current new request being inserted
                    if s.request_id == request.request_id:
                        continue

                    orig_eta = orig_eta_map.get(id(s))
                    if orig_eta is None:
                        continue

                    delay = max(0.0, etas[i] - orig_eta)

                    r = request_lookup_by_res_id.get(s.request_id)
                    if r is not None:
                        # --- existing passenger pickup-wait feasibility ---
                        # only applies to passengers not yet picked up
                        # if s.stop_type == StopType.PICKUP:
                        #     req_max_wait_existing = getattr(r, "max_wait", max_wait)
                        #     pickup_wait = etas[i] - r.request_time
                        #     if pickup_wait > req_max_wait_existing:
                        #         _log(
                        #             f"Candidate rejected BECAUSE EXISTING REQ WAIT VIOLATION: "
                        #             f"req={s.request_id} wait={pickup_wait:.1f}s > "
                        #             f"max_wait={req_max_wait_existing:.1f}s"
                        #         )
                        #         feasible = False
                        #         break
                        #     max_pickup_delay = max(pickup_wait, max_pickup_delay)

                        if s.stop_type == StopType.PICKUP:
                            req_max_wait_existing = getattr(r, "max_wait", max_wait)
                            pickup_wait = etas[i] - r.request_time
                            wait_violation = max(0.0, pickup_wait - req_max_wait_existing)

                            existing_wait_violation_sum += wait_violation
                            existing_wait_violation_max = max(existing_wait_violation_max, wait_violation)
                            max_pickup_delay = max(max_pickup_delay, pickup_wait)

                        # --- existing passenger ride-time feasibility ---
                        # elif s.stop_type == StopType.DROPOFF and r.direct_travel_time > 0:
                        #     pu_new_eta = next(
                        #         (etas[j] for j, ss in enumerate(new_stops)
                        #          if ss.request_id == s.request_id
                        #          and ss.stop_type == StopType.PICKUP),
                        #         None
                        #     )
                        #     if pu_new_eta is not None:
                        #         max_allowed_ride = (
                        #             getattr(r, "max_ride_factor", max_ride_factor)
                        #             * r.direct_travel_time
                        #         )
                        #         actual_ride = etas[i] - pu_new_eta
                        #         if actual_ride > max_allowed_ride:
                        #             _log(
                        #                 f"Candidate rejected BECAUSE EXISTING REQ RIDE VIOLATION: "
                        #                 f"req={s.request_id} ride={actual_ride:.1f}s > "
                        #                 f"limit={max_allowed_ride:.1f}s"
                        #             )
                        #             feasible = False
                        #             break

                        elif s.stop_type == StopType.DROPOFF and r.direct_travel_time > 0:
                            max_allowed_ride = (
                                getattr(r, "max_ride_factor", max_ride_factor) * r.direct_travel_time
                            )

                            # Case 1: passenger not yet picked up in this candidate plan
                            pu_new_eta = next(
                                (etas[j] for j, ss in enumerate(new_stops)
                                if ss.request_id == s.request_id
                                and ss.stop_type == StopType.PICKUP),
                                None
                            )

                            if pu_new_eta is not None:
                                actual_ride_existing = etas[i] - pu_new_eta
                                ride_violation = max(0.0, actual_ride_existing - max_allowed_ride)

                            # Case 2: passenger already onboard, pickup stop no longer remains
                            elif getattr(r, "pickup_time", None) is not None:
                                actual_ride_existing = etas[i] - r.pickup_time
                                ride_violation = max(0.0, actual_ride_existing - max_allowed_ride)

                            # Case 3: fallback if pickup_time is missing for some reason
                            # At least penalize the extra delay added to the dropoff itself.
                            else:
                                old_dropoff_eta = orig_eta
                                ride_violation = max(0.0, etas[i] - old_dropoff_eta)

                            existing_ride_violation_sum += ride_violation
                            existing_ride_violation_max = max(existing_ride_violation_max, ride_violation)

                    max_existing_delay = max(max_existing_delay, delay)
                    total_existing_delay += delay
                    existing_count += 1

                # if not feasible:
                #     continue

                avg_existing_delay = (
                    total_existing_delay / existing_count
                    if existing_count else 0.0
                )

                # measure the extra in vehicle time for passengers
                added_route_time = _compute_added_existing_passenger_ride_time(
                    plan=plan,
                    new_stops=new_stops,
                    orig_etas=orig_etas,
                    new_etas=etas,
                    now=now,
                )

                # Write computed ETAs back onto the Stop objects so display
                # shows real values instead of the default 0.0
                for stop_obj, eta_val in zip(new_stops, etas):
                    stop_obj.eta = eta_val

                c = CandidateInsertion(
                    request_id=request.request_id,
                    taxi_id=taxi_id,
                    pickup_index=pu_idx,
                    dropoff_index=do_idx,
                    resulting_stops=new_stops,
                    added_route_time=added_route_time,
                    pickup_eta_new=pu_eta,
                    dropoff_eta_new=do_eta,
                    max_existing_delay=max_existing_delay,
                    # can consider not using avg existing delay because it could be represented by 
                    # added_route_time ady
                    avg_existing_delay=avg_existing_delay,
                    max_pickup_delay=max_pickup_delay,
                    new_wait_violation=new_wait_violation,
                    new_ride_violation=new_ride_violation,
                    existing_wait_violation_sum=existing_wait_violation_sum,
                    existing_wait_violation_max=existing_wait_violation_max,
                    existing_ride_violation_sum=existing_ride_violation_sum,
                    existing_ride_violation_max=existing_ride_violation_max,
                    is_feasible=True,
                )
                candidates.append(c)

    # ── Per-taxi candidate capping ──────────────────────────────────
    # A taxi with N existing stops generates O(N²) insertion candidates.
    # An idle taxi generates exactly 1. Without capping, the candidate
    # set is heavily skewed toward busy taxis, biasing both random
    # exploration and greedy action selection.
    #
    # Fix: keep only the top-K candidates per taxi, ranked by a quick
    # heuristic (low added route time + low pickup ETA = best).
    # This bounds the total set to ≤ K*num_taxis + 1 (defer).
    MAX_CANDIDATES_PER_TAXI = 5
    from collections import defaultdict
    taxi_groups: dict[str, list[CandidateInsertion]] = defaultdict(list)
    for c in candidates:
        taxi_groups[c.taxi_id].append(c)

    capped: list[CandidateInsertion] = []
    for taxi_id, group in taxi_groups.items():
        if len(group) <= MAX_CANDIDATES_PER_TAXI:
            capped.extend(group)
        else:
            # Quick rank: prefer candidates with low added route time
            # and early pickup (i.e., short wait for the new passenger)
            group.sort(key=lambda c: c.added_route_time + 0.5 * (c.pickup_eta_new - now))
            capped.extend(group[:MAX_CANDIDATES_PER_TAXI])
    candidates = capped

    # always include DEFER
    candidates.append(CandidateInsertion.make_defer(request.request_id))
    return candidates


def _compute_added_existing_passenger_ride_time(
    plan: TaxiPlan,
    new_stops: list[Stop],
    orig_etas: list[float],
    new_etas: list[float],
    now: float,
) -> float:
    """
    Sum the additional in-vehicle time imposed on EXISTING passengers only.

    For a passenger who still has both PICKUP and DROPOFF in the plan:
        delta = (new_dropoff - new_pickup) - (old_dropoff - old_pickup)

    For a passenger already onboard (no PICKUP stop remaining in old plan):
        delta = (new_dropoff - now) - (old_dropoff - now)
              = new_dropoff - old_dropoff

    The new request being inserted should not be included here.
    """
    def _pickup_dropoff_eta_maps(stops: list[Stop], etas: list[float]):
        pu_map: dict[str, float] = {}
        do_map: dict[str, float] = {}
        for s, eta in zip(stops, etas):
            if s.stop_type == StopType.PICKUP:
                pu_map[s.request_id] = eta
            elif s.stop_type == StopType.DROPOFF:
                do_map[s.request_id] = eta
        return pu_map, do_map

    old_pu, old_do = _pickup_dropoff_eta_maps(plan.stops, orig_etas)
    new_pu, new_do = _pickup_dropoff_eta_maps(new_stops, new_etas)

    existing_request_ids = {
        s.request_id for s in plan.stops
    }

    total_added = 0.0

    for rid in existing_request_ids:
        if rid not in old_do or rid not in new_do:
            continue

        old_has_pickup = rid in old_pu
        new_has_pickup = rid in new_pu

        # Case 1: not yet picked up in both old and new plans
        if old_has_pickup and new_has_pickup:
            old_ride = old_do[rid] - old_pu[rid]
            new_ride = new_do[rid] - new_pu[rid]
            total_added += max(0.0, new_ride - old_ride)

        # Case 2: already onboard in old and still onboard in new
        elif not old_has_pickup and not new_has_pickup:
            old_remaining = old_do[rid] - now
            new_remaining = new_do[rid] - now
            total_added += max(0.0, new_remaining - old_remaining)

        # Case 3: fallback for mixed/edge situations
        # This should rarely happen, but if it does, compare dropoff ETA shift.
        else:
            total_added += max(0.0, new_do[rid] - old_do[rid])

    return total_added

class OnlineScoreNormalizer:
    """
    Running z-score normalizer for score components.

    Uses Welford's online algorithm:
      mean_n
      variance_n
    so we can normalize each metric without needing a separate preprocessing run.
    """

    def __init__(self, log_scale_keys: set[str] | None = None, clip_value: float | None = None):
        self.stats: dict[str, dict[str, float]] = {}
        self.log_scale_keys = log_scale_keys or set()
        self.clip_value = clip_value

    def _transform(self, key: str, value: float) -> float:
        v = float(value)

        # compress heavy-tailed positive costs
        if key in self.log_scale_keys:
            v = math.log1p(max(0.0, v))

        return v

    def update_and_normalize(self, key: str, value: float) -> float:
        """
        Transform -> update running stats -> return z-score.
        If too few samples, return transformed value directly at first,
        then gradually transition to z-scores.
        """
        x = self._transform(key, value)

        s = self.stats.setdefault(
            key,
            {"n": 0.0, "mean": 0.0, "M2": 0.0}
        )

        s["n"] += 1.0
        n = s["n"]

        delta = x - s["mean"]
        s["mean"] += delta / n
        delta2 = x - s["mean"]
        s["M2"] += delta * delta2

        # warm-up phase: avoid unstable normalization when sample count is tiny
        if n < 5:
            z = x
        else:
            variance = s["M2"] / max(1.0, n - 1.0)
            std = math.sqrt(max(variance, 1e-6))
            z = (x - s["mean"]) / std

        if self.clip_value is not None:
            z = max(-self.clip_value, min(self.clip_value, z))

        return z

    def get_summary_rows(self) -> list[dict]:
        """
        Export stats for debugging / CSV if needed.
        """
        rows = []
        for key, s in sorted(self.stats.items()):
            n = s["n"]
            variance = s["M2"] / max(1.0, n - 1.0) if n > 1 else 0.0
            std = math.sqrt(max(variance, 0.0))
            rows.append({
                "metric": key,
                "count": int(n),
                "mean": s["mean"],
                "std": std,
            })
        return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _unique_req_ids_in_stops(stops: list[Stop]) -> set[str]:
    return {s.request_id for s in stops}


def _plan_remaining_workload(plan: TaxiPlan, now: float) -> float:
    if not plan.stops:
        return 0.0
    last_eta = getattr(plan.stops[-1], "eta", 0.0)
    if last_eta and last_eta > 0:
        return max(0.0, last_eta - now)
    return max(0.0, getattr(plan, "remaining_route_time", 0.0))


def _normalize_component_dict(raw_components: dict[str, float]) -> dict[str, float]:
    """
    Normalize component costs so different terms live on a comparable scale.
    """
    if not NORMALIZE_SCORE_COMPONENTS:
        return dict(raw_components)

    normed: dict[str, float] = {}
    for key, value in raw_components.items():
        normed[key] = _SCORE_NORMALIZER.update_and_normalize(key, value)
    return normed


def _append_score_metrics_row_with_both(raw_row: dict, norm_row: dict) -> None:
    """
    Save both raw and normalized metrics for analysis.
    """
    merged = {}

    for k, v in raw_row.items():
        merged[f"raw_{k}"] = v

    for k, v in norm_row.items():
        merged[f"norm_{k}"] = v

    _append_score_metrics_row(merged)


def score_candidate(
    c: CandidateInsertion,
    request: Request,
    taxi_plans: dict[str, TaxiPlan],
    now: float,
) -> float:
    """
    Scoring pipeline:
      1. compute raw component metrics
      2. normalize raw metrics online
      3. apply weights AFTER normalization
      4. combine into final score
    """
    if c.is_defer:
        return -1e9

    plan = taxi_plans.get(c.taxi_id)
    if plan is None:
        return -1e9

    waiting_so_far = max(0.0, now - request.request_time)
    future_wait = max(0.0, c.pickup_eta_new - now)
    total_wait = max(0.0, c.pickup_eta_new - request.request_time)

    is_idle_activation = (plan.status == TaxiStatus.IDLE and len(plan.stops) == 0)
    slack = max(0.0, getattr(request, "max_wait", 300.0) - total_wait)
    urgency_factor = 1.0 if slack >= 60.0 else max(0.10, slack / 60.0)
    activation_raw = urgency_factor if is_idle_activation else 0.0

    suffix_completion = (
        max(0.0, c.resulting_stops[-1].eta - now)
        if c.resulting_stops else 0.0
    )

    other_workloads = [
        _plan_remaining_workload(tp, now)
        for tid, tp in taxi_plans.items()
        if tid != c.taxi_id
    ]
    baseline_other = min(other_workloads) if other_workloads else 0.0
    imbalance_raw = max(0.0, suffix_completion - baseline_other)

    before_unique = len(_unique_req_ids_in_stops(plan.stops)) + len(plan.onboard_request_ids)
    after_unique = len(_unique_req_ids_in_stops(c.resulting_stops)) + len(plan.onboard_request_ids)
    share_gain = max(0, after_unique - max(1, before_unique))
    compact_share = (
        after_unique >= 2
        and c.max_existing_delay <= 30.0
        and future_wait <= 150.0
    )
    share_raw = float(share_gain) if compact_share else 0.0

    # --------------------------------------------------
    # Raw metrics only (NO weights here)
    # --------------------------------------------------
    raw_cost_components = {
        "max_delay": c.max_existing_delay,
        "avg_delay": c.avg_existing_delay,
        "waited": waiting_so_far,
        "future_wait": future_wait,
        "total_wait": total_wait,
        "route": c.added_route_time,
        "activation": activation_raw,
        "workload": suffix_completion,
        "imbalance": imbalance_raw,
        "new_wait_viol": c.new_wait_violation ** 2,
        "new_ride_viol": c.new_ride_violation ** 2,
        "exist_wait_viol": (
            (c.existing_wait_violation_sum ** 2)
            + 2.0 * (c.existing_wait_violation_max ** 2)
        ),
        "exist_ride_viol": (
            (c.existing_ride_violation_sum ** 2)
            + 2.0 * (c.existing_ride_violation_max ** 2)
        ),
    }

    raw_bonus_components = {
        "share": share_raw,
    }

    # --------------------------------------------------
    # Normalize raw metrics first
    # --------------------------------------------------
    norm_cost_components = _normalize_component_dict(raw_cost_components)
    norm_bonus_components = _normalize_component_dict(raw_bonus_components)

    # --------------------------------------------------
    # Apply weights AFTER normalization
    # --------------------------------------------------
    weighted_cost_components = {
        "max_delay_cost": W_MAX_ONBOARD_DELAY * norm_cost_components["max_delay"],
        # "avg_delay_cost": W_AVG_ONBOARD_DELAY * norm_cost_components["avg_delay"],
        "waited_cost": W_WAIT_SO_FAR * norm_cost_components["waited"],
        "future_wait_cost": W_FUTURE_WAIT * norm_cost_components["future_wait"],
        # "total_wait_cost": W_TOTAL_WAIT * norm_cost_components["total_wait"],
        "route_cost": W_ROUTE * norm_cost_components["route"],
        "activation_cost": W_ACTIVATION * norm_cost_components["activation"],
        "workload_cost": W_WORKLOAD * norm_cost_components["workload"],
        "imbalance_cost": W_IMBALANCE * norm_cost_components["imbalance"],
        "new_wait_viol_cost": W_NEW_WAIT_VIOL * norm_cost_components["new_wait_viol"],
        "new_ride_viol_cost": W_NEW_RIDE_VIOL * norm_cost_components["new_ride_viol"],
        "exist_wait_viol_cost": W_EXIST_WAIT_VIOL * norm_cost_components["exist_wait_viol"],
        "exist_ride_viol_cost": W_EXIST_RIDE_VIOL * norm_cost_components["exist_ride_viol"],
    }

    weighted_bonus_components = {
        "share_bonus": W_SHARE_BONUS * norm_bonus_components["share"],
    }

    # --------------------------------------------------
    # Save raw / normalized / weighted values
    # --------------------------------------------------
    debug_row = {}

    for k, v in raw_cost_components.items():
        debug_row[f"raw_{k}"] = v
    for k, v in raw_bonus_components.items():
        debug_row[f"raw_{k}"] = v

    for k, v in norm_cost_components.items():
        debug_row[f"norm_{k}"] = v
    for k, v in norm_bonus_components.items():
        debug_row[f"norm_{k}"] = v

    for k, v in weighted_cost_components.items():
        debug_row[k] = v
    for k, v in weighted_bonus_components.items():
        debug_row[k] = v

    _append_score_metrics_row(debug_row)

    total_cost = sum(weighted_cost_components.values()) - sum(weighted_bonus_components.values())
    return -total_cost


SCORE_RECORD_FILE = Path("score_metrics_record.csv")
_SCORE_RECORD_HEADER_WRITTEN = False


_SCORE_NORMALIZER = OnlineScoreNormalizer(
    log_scale_keys=LOG_SCALE_KEYS,
    clip_value=NORMALIZED_CLIP_VALUE,
)

# this is a helper function to record the metrics into csv
def _append_score_metrics_row(row: dict) -> None:
    """
    Append one candidate-scoring record into a CSV file.
    Writes header once per run if the file is new/empty.
    """
    global _SCORE_RECORD_HEADER_WRITTEN

    try:
        file_exists_and_nonempty = SCORE_RECORD_FILE.exists() and SCORE_RECORD_FILE.stat().st_size > 0

        with SCORE_RECORD_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            if not _SCORE_RECORD_HEADER_WRITTEN and not file_exists_and_nonempty:
                writer.writeheader()
                _SCORE_RECORD_HEADER_WRITTEN = True
            elif not _SCORE_RECORD_HEADER_WRITTEN:
                # file already has content from previous run
                _SCORE_RECORD_HEADER_WRITTEN = True

            writer.writerow(row)

    except Exception as e:
        _log(f"[WARN] failed to append score metrics CSV: {e}")

# ---------------------------------------------------------------------------
# State tracking helpers
# ---------------------------------------------------------------------------

def _sumo_taxi_status(fleet_state: int) -> TaxiStatus:
    return {
        0: TaxiStatus.IDLE,
        1: TaxiStatus.PICKUP,
        2: TaxiStatus.OCCUPIED,
        3: TaxiStatus.PICKUP_AND_OCCUPIED,
    }.get(fleet_state, TaxiStatus.IDLE)


def _refresh_taxi_plans(taxi_plans: dict[str, TaxiPlan]) -> None:
    """Pull current position and status for every known taxi from SUMO."""
    active_ids = set(traci.vehicle.getIDList())
    for taxi_id, plan in taxi_plans.items():
        if taxi_id not in active_ids:
            continue   # taxi left the sim — skip silently
        try:
            plan.current_edge = traci.vehicle.getRoadID(taxi_id)
            # x, y = traci.vehicle.getPosition(taxi_id)
            # plan.current_x = x
            # plan.current_y = y
            # _log(f"Updated taxi {taxi_id} position: {plan.current_edge} x={x:.1f} y={y:.1f}")
        except traci.TraCIException:
            pass

    # update statuses from fleet query
    for state_int in range(4):
        for vid in traci.vehicle.getTaxiFleet(state_int):
            if vid in taxi_plans:
                taxi_plans[vid].status = _sumo_taxi_status(state_int)


def _sync_onboard(
    taxi_plans: dict[str, TaxiPlan],
    requests: dict[str, Request],
) -> None:
    """
    Sync onboard_request_ids and onboard_count for each taxi.
    Uses traci.vehicle.getPersonIDList() for a ground-truth count directly
    from SUMO rather than relying on our status bookkeeping, which can lag.
    """
    active_ids = set(traci.vehicle.getIDList())
    for taxi_id, plan in taxi_plans.items():
        if taxi_id not in active_ids:
            plan.onboard_count = 0   # taxi gone, no one onboard
            continue
        try:
            persons_in_taxi = set(traci.vehicle.getPersonIDList(taxi_id))
            # _log(f"[PERSON IN TAXI] {taxi_id} {persons_in_taxi}")
        except traci.TraCIException:
            persons_in_taxi = set()
        # intersect with requests we know about to get tracked onboard set
        plan.onboard_request_ids = {
            pid for pid in persons_in_taxi if pid in requests
        }
        plan.onboard_count = len(plan.onboard_request_ids)
        # also keep assigned_request_ids consistent — remove completed ones
        plan.assigned_request_ids = {
            rid for rid in plan.assigned_request_ids
            if requests.get(rid) and requests[rid].status != RequestStatus.COMPLETED
        }


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _detect_events(
    prev_known_req_ids: set[str],
    prev_onboard_ids:   set[str],
    prev_completed_ids: set[str],
    requests: dict[str, Request],
) -> tuple[bool, list[str], list[str], list[str]]:
    """
    Compare current state against previous snapshot to find meaningful events.

    Returns:
        had_event      : True if any meaningful event occurred
        new_arrivals   : person_ids of newly appeared requests
        new_pickups    : person_ids just transitioned to ONBOARD
        new_dropoffs   : person_ids just transitioned to COMPLETED
    """
    current_req_ids  = set(requests.keys())
    current_onboard  = {pid for pid, r in requests.items() if r.status == RequestStatus.ONBOARD}
    current_complete = {pid for pid, r in requests.items() if r.status == RequestStatus.COMPLETED}

    new_arrivals  = list(current_req_ids  - prev_known_req_ids)
    new_pickups   = list(current_onboard  - prev_onboard_ids)
    new_dropoffs  = list(current_complete - prev_completed_ids)
    had_event = bool(new_arrivals or new_pickups or new_dropoffs)
    if had_event:
        _log(f"New arrivals: {new_arrivals}, pickups: {new_pickups}, dropoffs: {new_dropoffs}")
    return had_event, new_arrivals, new_pickups, new_dropoffs


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_top5(
    candidates: list[CandidateInsertion],
    scores: list[float],
    request: Request,
    taxi_plans: dict[str, TaxiPlan],
    now: float,
) -> None:
    paired = sorted(zip(scores, candidates), key=lambda x: -x[0])
    _log(f"\n  ┌─ TOP-5 CANDIDATES for request {request.request_id}"
         f" (person {request.person_id})"
         f" waited {request.waiting_time(now):.1f}s ─┐")
    for rank, (sc, c) in enumerate(paired[:5], 1):
        if c.is_defer:
            _log(f"  │  #{rank}  DEFER                              score={sc:+.2f}")
        else:
            plan = taxi_plans[c.taxi_id]
            idle_tag = " [ACTIVATES IDLE TAXI]" if (
                plan.status == TaxiStatus.IDLE and len(plan.stops) == 0) else ""
            _log(
                f"  │  #{rank}  taxi={c.taxi_id:>6}  "
                f"pu_idx={c.pickup_index}  do_idx={c.dropoff_index}  "
                f"+route={c.added_route_time:+6.1f}s  "
                f"max_delay={c.max_existing_delay:5.1f}s  "
                f"pu_eta={c.pickup_eta_new:7.1f}s"
                f"{idle_tag}  "
                f"score={sc:+.2f}"
            )
    _log("  └" + "─" * 70)


def _print_tick_summary(
    tick_num: int,
    sim_time: float,
    tick: TickContext,
    new_arrivals: list[str],
    new_pickups: list[str],
    new_dropoffs: list[str],
    requests: dict[str, Request],
    taxi_plans: dict[str, TaxiPlan],
    accumulator: IntervalAccumulator,
) -> None:
    pending_reqs = [r for r in requests.values()
                    if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
    onboard_reqs = [r for r in requests.values() if r.status == RequestStatus.ONBOARD]
    completed    = [r for r in requests.values() if r.status == RequestStatus.COMPLETED]
    assigned     = [r for r in requests.values() if r.status == RequestStatus.ASSIGNED]

    active_taxis = [p for p in taxi_plans.values() if p.status != TaxiStatus.IDLE or p.stops]
    idle_taxis   = [p for p in taxi_plans.values() if p.status == TaxiStatus.IDLE and not p.stops]

    avg_wait = (
        sum(r.waiting_time(sim_time) for r in pending_reqs) / len(pending_reqs)
        if pending_reqs else 0.0
    )
    max_wait = (
        max(r.waiting_time(sim_time) for r in pending_reqs)
        if pending_reqs else 0.0
    )

    outcome_tag = f"[{tick.outcome.name}]"
    sep = "═" * 72
    _log(f"\n{sep}")
    _log(f"  TICK #{tick_num:04d}  t={sim_time:8.1f}s  {outcome_tag}")
    _log(f"  Events this tick:")
    if new_arrivals:
        _log(f"    + Arrivals  : {new_arrivals}")
    if new_pickups:
        _log(f"    ✓ Pickups   : {new_pickups}")
    if new_dropoffs:
        _log(f"    ✗ Dropoffs  : {new_dropoffs}")
    if not (new_arrivals or new_pickups or new_dropoffs):
        _log(f"    (none)")
    _log(f"  Requests  : pending={len(pending_reqs)}  assigned={len(assigned)}"
         f"  onboard={len(onboard_reqs)}  completed={len(completed)}")
    if pending_reqs:
        _log(f"  Wait times: avg={avg_wait:.1f}s  max={max_wait:.1f}s")
    _log(f"  Fleet     : active={len(active_taxis)}  idle={len(idle_taxis)}"
         f"  total={len(taxi_plans)}")
    for plan in sorted(taxi_plans.values(), key=lambda p: p.taxi_id):
        stops_str = "  ".join(repr(s) for s in plan.stops) if plan.stops else "(no stops)"
        _log(f"    taxi {plan.taxi_id:>6}: {plan.status.name:<22} load={plan.onboard_count}"
             f"  stops=[{stops_str}]")
    _log(f"  Interval accumulator: {accumulator}")
    _log(sep)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

class HeuristicDispatcher:
    def __init__(self, cfg_path: str, step_length: float = 1.0, use_gui: bool = False):
        self.cfg_path    = cfg_path
        self.step_length = step_length
        self.use_gui     = use_gui

        self.requests:    dict[str, Request]  = {}   # person_id  → Request
        self.taxi_plans:  dict[str, TaxiPlan] = {}   # taxi_id    → TaxiPlan
        self.pid_to_resid: dict[str, str]     = {}   # person_id  → reservation_id (res.id)
        self.resid_to_pid: dict[str, str]     = {}   # reservation_id → person_id
        self._pending_dispatches: set[str]    = set() # taxi_ids needing dispatchTaxi this tick
        # taxi_id -> snapshot taken before the first mutation this tick
        # used to rollback only this tick's tentative re-plan if dispatchTaxi fails
        self._dispatch_snapshots: dict[str, tuple[list[Stop], set[str]]] = {}

        self.accumulator = IntervalAccumulator()

        # event detection snapshots
        self._prev_req_ids:       set[str] = set()
        self._prev_onboard_ids:   set[str] = set()
        self._prev_completed_ids: set[str] = set()

        self._tick_num    = 0
        self._step_count  = 0   # steps since last tick
        self.TICK_STEPS   = 10  # how many sim steps per tick
        self._eligible_taxis_this_tick: set[str] = set()
        self._cached_vtype_str: str = ""  # cached vehicle type string for route calculation
        # Metadata for re-creating taxis that SUMO removes after they go idle.
        # Stores {taxi_id: {"vtype": str, "route_id": str, "capacity": int}}
        self._taxi_metadata: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([binary, "-c", self.cfg_path,
                     "--step-length", str(self.step_length)])
        self._init_taxi_plans()

    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass   # connection may already be closed (e.g. after KeyboardInterrupt)

    def _init_taxi_plans(self) -> None:
        """Register all taxis already in the simulation at startup."""
        for vid in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(vid).startswith("taxi") or \
                   "taxi" in traci.vehicle.getVehicleClass(vid).lower():
                    self._register_taxi(vid)
                    print(vid)
            except traci.TraCIException:
                pass
            
        # also probe via fleet API
        for state_int in range(4):
            for vid in traci.vehicle.getTaxiFleet(state_int):
                if vid not in self.taxi_plans:
                    self._register_taxi(vid)

    def _register_taxi(self, taxi_id: str) -> None:
        # If taxi already exists, DO NOT wipe its stops / assignments.
        existing = self.taxi_plans.get(taxi_id)
        if existing is not None:
            try:
                existing.current_edge = traci.vehicle.getRoadID(taxi_id)
                SAFETY_BUFFER = 3
                real_capacity = int(traci.vehicle.getPersonCapacity(taxi_id))
                existing.capacity = max(1, real_capacity - SAFETY_BUFFER)
            except traci.TraCIException:
                pass
            _log(
                f"  [INIT-SKIP] Taxi {taxi_id} already registered; "
                f"preserving existing plan with {len(existing.stops)} stops"
            )
            return

        plan = TaxiPlan(taxi_id=taxi_id)
        try:
            plan.current_edge = traci.vehicle.getRoadID(taxi_id)
            SAFETY_BUFFER = 3
            real_capacity = int(traci.vehicle.getPersonCapacity(taxi_id))
            plan.capacity = max(1, real_capacity - SAFETY_BUFFER)
            _log(f"CAPACITY OF TAXI: {plan.capacity}")
            # Cache the vehicle type string for route calculations.
            # This avoids needing a live taxi for getTypeID() later.
            vtype_id = traci.vehicle.getTypeID(taxi_id)
            if not self._cached_vtype_str:
                self._cached_vtype_str = vtype_id
            # Store metadata so we can re-create this taxi if SUMO removes it
            try:
                route_id = traci.vehicle.getRouteID(taxi_id)
            except Exception:
                route_id = ""
            self._taxi_metadata[taxi_id] = {
                "vtype": vtype_id,
                "route_id": route_id,
                "capacity": plan.capacity,
                "initial_edge": plan.current_edge,
            }
        except traci.TraCIException:
            plan.capacity = 2

        self.taxi_plans[taxi_id] = plan
        _log(f"  [INIT] Registered taxi {taxi_id} (capacity={plan.capacity})")

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def _revive_missing_taxis(self) -> None:
        """Re-add taxis that SUMO has removed after they completed all work.

        SUMO removes taxi vehicles from the simulation once they finish their
        last dispatched stop and have no remaining route. This is a problem
        for DRT because we want taxis to stay available for future requests.

        This method detects taxis in our local taxi_plans that are no longer
        in SUMO's vehicle list and re-adds them using traci.vehicle.add()
        with the same vehicle type. The taxi is placed on its last known edge.
        """
        try:
            active_vids = set(traci.vehicle.getIDList())
        except Exception:
            return

        for taxi_id, plan in list(self.taxi_plans.items()):
            if taxi_id in active_vids:
                continue

            # Only revive taxis that have no remaining work (stops/onboard).
            # If a taxi has stops but is missing, something else is wrong.
            if plan.stops or plan.onboard_count > 0:
                continue

            meta = self._taxi_metadata.get(taxi_id)
            if not meta:
                continue

            vtype = meta["vtype"]
            # Use the last known edge as the departure edge
            depart_edge = plan.current_edge or meta.get("initial_edge", "")
            if not depart_edge or depart_edge.startswith(":"):
                # Internal junction edge — use the initial edge instead
                depart_edge = meta.get("initial_edge", "")
            if not depart_edge:
                _log(f"  [REVIVE-SKIP] taxi={taxi_id}: no valid edge to place taxi on")
                continue

            try:
                # Create a single-edge route for the taxi to depart on
                route_id = f"revive_route_{taxi_id}"
                try:
                    traci.route.add(route_id, [depart_edge])
                except traci.TraCIException:
                    # Route may already exist from a previous revive
                    pass

                traci.vehicle.add(
                    taxi_id,
                    route_id,
                    typeID=vtype,
                    departPos="random_free",
                    departSpeed="0",
                )
                # Reset the plan state for a fresh idle taxi
                plan.status = TaxiStatus.IDLE
                plan.cumulative_distance = 0.0
                # Re-read position after adding
                try:
                    plan.current_edge = traci.vehicle.getRoadID(taxi_id)
                except Exception:
                    plan.current_edge = depart_edge

                _log(f"  [REVIVE] Re-added taxi={taxi_id} on edge={depart_edge} (vtype={vtype})")
            except traci.TraCIException as e:
                _log(f"  [REVIVE-FAIL] Could not re-add taxi={taxi_id}: {e}")

    def _redispatch_remaining(self, taxi_id: str, plan: TaxiPlan) -> None:
        """Re-issue dispatchTaxi with the remaining stops after a pickup or
        dropoff so SUMO keeps routing to the next assigned passengers.

        Without this call SUMO may consider its dispatch obligation complete
        after finishing the last stop it was actively routing to, go IDLE,
        and silently cancel reservations for passengers still in our plan.

        Only reservation IDs that SUMO still recognises are included.
        Dead IDs are purged from the local plan and the affected requests
        are reset to PENDING so they re-enter the dispatch pool.
        """
        # Guard: verify taxi still exists in SUMO before any dispatch call
        try:
            active_vids = set(traci.vehicle.getIDList())
        except Exception:
            return
        if taxi_id not in active_vids:
            _log(f"  [REDISPATCH-SKIP] taxi={taxi_id} no longer in simulation")
            return

        remaining_res_ids = _serialize_dispatch_res_ids(plan)
        if not remaining_res_ids:
            return

        # Check which reservation IDs are still alive in SUMO
        try:
            live_res_ids = {r.id for r in traci.person.getTaxiReservations(0)}
        except Exception:
            live_res_ids = None

        if live_res_ids is not None:
            dead_rids = {rid for rid in remaining_res_ids if rid not in live_res_ids}
            if dead_rids:
                _log(f"  [REDISPATCH] purging dead reservation IDs from taxi={taxi_id}: {dead_rids}")
                for dead_rid in dead_rids:
                    plan.stops = [s for s in plan.stops if s.request_id != dead_rid]
                    pid = self.resid_to_pid.get(dead_rid, dead_rid)
                    plan.assigned_request_ids.discard(dead_rid)
                    plan.assigned_request_ids.discard(pid)
                    plan.onboard_request_ids.discard(dead_rid)
                    plan.onboard_request_ids.discard(pid)
                    req = self.requests.get(pid)
                    if req and req.status not in (RequestStatus.ONBOARD, RequestStatus.COMPLETED):
                        req.assigned_taxi_id = None
                        req.status = RequestStatus.PENDING
                        _log(f"    → {dead_rid} reservation dead — reset to PENDING for re-dispatch")
                plan.onboard_count = len(plan.onboard_request_ids)
                # Rebuild after purge
                remaining_res_ids = _serialize_dispatch_res_ids(plan)

        if not remaining_res_ids:
            return

        try:
            traci.vehicle.dispatchTaxi(taxi_id, remaining_res_ids)
            _log(f"  [REDISPATCH] taxi={taxi_id} reservations={remaining_res_ids}")
        except traci.TraCIException as e:
            _log(f"  [REDISPATCH-WARN] dispatchTaxi failed for taxi={taxi_id}: {e}")

    def _sync_reservations(self, now: float) -> None:
        """
        Sync all request and taxi state from SUMO.

        Strategy: scan all active persons directly via traci.person, and
        use getTaxiReservations() only to discover new requests and get
        their from/to edges. Status is determined by:

          PENDING/ASSIGNED : person exists in sim AND is NOT inside a vehicle
                             (traci.person.getVehicle(pid) == "")
          ONBOARD          : person exists AND traci.person.getVehicle(pid) != ""
          COMPLETED        : person no longer in traci.person.getIDList()

        This avoids any dependency on reservation.state bitmasks or
        stage.type values, both of which proved unreliable here.
        """
        # Track taxis whose plans were modified so we can re-dispatch
        # them ONCE at the end, rather than per-event inside the loop.
        self._taxis_needing_redispatch: set[str] = set()

        # --- revive taxis that SUMO removed after they went idle ---
        self._revive_missing_taxis()

        # --- lazy taxi registration ---
        for state_int in range(4):
            for vid in traci.vehicle.getTaxiFleet(state_int):
                if vid not in self.taxi_plans:
                    _log(f"  [LAZY-REGISTER] state={state_int} taxi={vid} known={vid in self.taxi_plans}")
                    self._register_taxi(vid)

        # --- discover new requests via reservations ---
        # getTaxiReservations(0) = return ALL reservations (new + assigned + onboard)
        # In some SUMO versions the filter arg behaves differently; we try both
        # 0 (all) and also walk persons directly as fallback.
        try:
            reservations = traci.person.getTaxiReservations(0)

        except Exception:
            reservations = []

        for res in reservations:
            pid = res.persons[0] if res.persons else None
            # skip if no person id, blank string, or already tracked
            if not pid or pid in self.requests:
                continue
            try:
                # Use cached vtype string to avoid calling getTypeID on a
                # taxi that may have been removed from SUMO after completing
                # all its work.
                vtype_str = self._cached_vtype_str
                if not vtype_str:
                    # Fallback: try to get vtype from a taxi still alive
                    try:
                        active_vids_for_vtype = set(traci.vehicle.getIDList())
                    except Exception:
                        active_vids_for_vtype = set()
                    alive_taxi = next(
                        (tid for tid in self.taxi_plans if tid in active_vids_for_vtype),
                        ""
                    )
                    if alive_taxi:
                        vtype_str = traci.vehicle.getTypeID(alive_taxi)
                        self._cached_vtype_str = vtype_str
                route = traci.simulation.findRoute(
                    res.fromEdge, res.toEdge, vtype_str, routingMode=1)
                dtt = route.travelTime
                max_wait = min(REQ_MAX_WAIT_CAP, REQ_BASE_WAIT + REQ_ALPHA * dtt)
            except Exception:
                print("Enter exception")
                dtt = 0.0
                max_wait = 0.0
            req = Request(
                request_id=res.id,
                person_id=pid,
                from_edge=res.fromEdge,
                to_edge=res.toEdge,
                request_time=now,
                direct_travel_time=dtt,
                max_wait=max_wait
            )
            self.requests[pid]     = req
            self.pid_to_resid[pid] = res.id   # map person→reservation for dispatchTaxi
            self.resid_to_pid[res.id] = pid
            _log(f"  [NEW REQUEST] person={pid}  res={res.id}"
                 f"  {res.fromEdge} → {res.toEdge}"
                 f"  direct_tt={dtt:.1f}s")

        # --- also discover persons not caught by reservations ---
        # (handles SUMO versions where getTaxiReservations returns empty)
        try:
            person_id_list = traci.person.getIDList()
        except traci.exceptions.FatalTraCIError:
            _log("  [WARN] TraCI connection closed — skipping person scan.")
            return
        for pid in person_id_list:
            if pid in self.requests:
                continue
            try:
                # only interested in persons with a taxi ride stage
                stages = [traci.person.getStage(pid, i)
                          for i in range(traci.person.getRemainingStages(pid))]
                ride = next((s for s in stages
                             if s.type == 3 or (hasattr(s, 'line') and s.line == 'taxi')), None)
                if ride is None:
                    continue
                from_edge = traci.person.getRoadID(pid)
                to_edge   = ride.edges[-1] if getattr(ride, "edges", None) else from_edge
                try:
                    # Use cached vtype string
                    vtype_str = self._cached_vtype_str
                    if not vtype_str:
                        try:
                            active_vids_for_vtype = set(traci.vehicle.getIDList())
                        except Exception:
                            active_vids_for_vtype = set()
                        alive_taxi = next(
                            (tid for tid in self.taxi_plans if tid in active_vids_for_vtype),
                            ""
                        )
                        if alive_taxi:
                            vtype_str = traci.vehicle.getTypeID(alive_taxi)
                            self._cached_vtype_str = vtype_str
                    route = traci.simulation.findRoute(from_edge, to_edge, vtype_str, routingMode=1)
                    dtt = route.travelTime
                    max_wait = min(REQ_MAX_WAIT_CAP, REQ_BASE_WAIT + REQ_ALPHA * dtt)
                except Exception:
                    dtt = 0.0
                    max_wait = 0.0 
                req = Request(
                    request_id=pid,   # use pid as fallback res id
                    person_id=pid,
                    from_edge=from_edge,
                    to_edge=to_edge,
                    request_time=now,
                    direct_travel_time=dtt,
                    max_wait=max_wait
                )
                self.requests[pid] = req
                self.pid_to_resid[pid] = req.request_id
                self.resid_to_pid[req.request_id] = pid
                _log(f"  [NEW REQUEST via person scan] person={pid}"
                     f"  {from_edge} → {to_edge}  direct_tt={dtt:.1f}s")
            except Exception:
                pass

        # --- update status for all tracked requests ---
        active_pids = set(traci.person.getIDList())
        # _log(f"ACTIVE PIDS: {active_pids}")

        for pid, req in self.requests.items():
            if req.status == RequestStatus.COMPLETED:
                continue

            in_sim = pid in active_pids

            if not in_sim:
                # person left simulation → dropped off
                if req.status == RequestStatus.ONBOARD:
                    req.status    = RequestStatus.COMPLETED
                    req.dropoff_time = now
                    if req.assigned_taxi_id and req.assigned_taxi_id in self.taxi_plans:
                        plan = self.taxi_plans[req.assigned_taxi_id]
                        plan.stops = [s for s in plan.stops
                                      if s.request_id != req.request_id]
                        plan.assigned_request_ids.discard(pid)
                        plan.onboard_request_ids.discard(pid)
                        plan.onboard_count = len(plan.onboard_request_ids)
                        # Mark for re-dispatch at end of _sync_reservations
                        self._taxis_needing_redispatch.add(req.assigned_taxi_id)
                    _log(f"  [DROPOFF] person={pid}  t={now:.1f}s"
                         f"  ride_time={(req.dropoff_time - req.pickup_time):.1f}s"
                         f"  excess={req.excess_ride_time}")
                else:
                    # person left without being tracked as onboard — mark done
                    # _log("CLEARUR UNKNOWN REQ") luckily this not happen for SmallTestingMap
                    req.status = RequestStatus.COMPLETED
                    req.dropoff_time = now
                    if req.assigned_taxi_id and req.assigned_taxi_id in self.taxi_plans:
                        plan = self.taxi_plans[req.assigned_taxi_id]
                        plan.stops = [s for s in plan.stops
                                      if s.request_id != req.request_id]
                        plan.assigned_request_ids.discard(pid)
                        plan.onboard_request_ids.discard(pid)
                        plan.onboard_count = len(plan.onboard_request_ids)
                        self._taxis_needing_redispatch.add(req.assigned_taxi_id)
                    _log(f"  [DROPOFF] person={pid}  t={now:.1f}s (left sim, was {req.status.name})")
                continue

            # person is still in sim — check if inside a vehicle
            try:
                in_vehicle = traci.person.getVehicle(pid) != ""
            except Exception:
                in_vehicle = False

            if in_vehicle and req.status not in (RequestStatus.ONBOARD, RequestStatus.COMPLETED):
                # just got picked up — transition to ONBOARD
                req.status      = RequestStatus.ONBOARD
                req.pickup_time = now
                vehicle_id = traci.person.getVehicle(pid)
                if vehicle_id and vehicle_id in self.taxi_plans:
                    req.assigned_taxi_id = vehicle_id
                    taxi_plan = self.taxi_plans[vehicle_id]
                    taxi_plan.assigned_request_ids.add(pid)
                    taxi_plan.onboard_request_ids.add(pid)
                    taxi_plan.onboard_count = len(taxi_plan.onboard_request_ids)
                    # Remove the PICKUP stop from our plan — SUMO has consumed it.
                    # Keeping it would cause generate_candidates to re-include it
                    # in resulting_stops, leading to dispatchTaxi duplicate errors.
                    taxi_plan.stops = [
                        s for s in taxi_plan.stops
                        if not (s.request_id == req.request_id
                                and s.stop_type == StopType.PICKUP)
                    ]
                    # Mark for re-dispatch at end of _sync_reservations
                    self._taxis_needing_redispatch.add(vehicle_id)
                _log(f"  [PERIODIC PICKUP] person={pid}  taxi={req.assigned_taxi_id}  t={now:.1f}s")

            elif not in_vehicle and req.status == RequestStatus.ONBOARD:
                # Transient state can happen right around a stop boundary.
                # Do not reset all the way back to PENDING, which can create
                # duplicate pickup insertions. Leave the request assigned and
                # let the next tick resolve it via normal SUMO state sync.
                req.status = RequestStatus.ASSIGNED
                _log(f"  [WARN] person={pid} left vehicle but still in sim — marking ASSIGNED transiently")

        # ── Batched re-dispatch ──────────────────────────────────────────
        # Issue ONE dispatchTaxi call per taxi that had stops modified
        # during this sync pass (pickup consumed, dropoff completed, etc.).
        # This prevents SUMO from going idle and cancelling remaining
        # reservations, while avoiding the N-calls-per-N-events storm
        # that can overwhelm SUMO and cause vehicle teleportation/removal.
        if hasattr(self, '_taxis_needing_redispatch'):
            try:
                active_vids = set(traci.vehicle.getIDList())
            except Exception:
                active_vids = set()

            for taxi_id in self._taxis_needing_redispatch:
                if taxi_id not in active_vids:
                    continue
                plan = self.taxi_plans.get(taxi_id)
                if plan and plan.stops:
                    self._redispatch_remaining(taxi_id, plan)
            self._taxis_needing_redispatch.clear()

    # ------------------------------------------------------------------
    # Dispatch logic
    # ------------------------------------------------------------------



    def _dispatch_best(self, request: Request, now: float) -> Optional[CandidateInsertion]:
        """
        Score all candidates for `request` across idle taxis, print top-5,
        and apply the best one. Does NOT call dispatchTaxi yet — the actual
        SUMO call is deferred to _flush_idle_dispatches so that all requests
        assigned to the same taxi in one tick are batched into a single call.
        """
        request_lookup = _build_request_lookup_by_res_id(self.requests)
        eligible_taxis = getattr(self, "_eligible_taxis_this_tick", set())
        
        # only open this one if needed
        # raw_candidates = enumerate_all_raw_candidates(
        #     request,
        #     self.taxi_plans,
        #     now,
        #     eligible_taxi_ids=eligible_taxis,
        # )
        # _print_all_raw_candidates(raw_candidates, request, now)

        candidates = generate_candidates(
            request, self.taxi_plans, self.requests, now,
            eligible_taxi_ids=eligible_taxis,
            request_lookup_by_res_id=request_lookup,
        )
        if not candidates:
            return None

        scores = [score_candidate(c, request, self.taxi_plans, now) for c in candidates]

        # Candidate-set-aware refinements: if an idle taxi can pick up materially
        # earlier than any already-busy taxi, give it an explicit bonus so we do
        # not keep overloading the busy taxi just to avoid a tiny activation cost.
        # Also penalise busy options that are clearly slower than the best idle one.
        # non_defer = [c for c in candidates if not c.is_defer]
        # busy_pickups = [
        #     cand.pickup_eta_new
        #     for cand in non_defer
        #     if not (
        #         self.taxi_plans.get(cand.taxi_id)
        #         and self.taxi_plans[cand.taxi_id].status == TaxiStatus.IDLE
        #         and len(self.taxi_plans[cand.taxi_id].stops) == 0
        #     )
        # ]
        # best_busy_pickup = min(busy_pickups) if busy_pickups else None
        # idle_pickups = [
        #     cand.pickup_eta_new
        #     for cand in non_defer
        #     if (
        #         self.taxi_plans.get(cand.taxi_id)
        #         and self.taxi_plans[cand.taxi_id].status == TaxiStatus.IDLE
        #         and len(self.taxi_plans[cand.taxi_id].stops) == 0
        #     )
        # ]
        # best_idle_pickup = min(idle_pickups) if idle_pickups else None
        # for i, cand in enumerate(candidates):
        #     if cand.is_defer:
        #         continue
        #     plan = self.taxi_plans.get(cand.taxi_id)
        #     if plan is None:
        #         continue
        #     is_idle = (plan.status == TaxiStatus.IDLE and len(plan.stops) == 0)
        #     if is_idle and best_busy_pickup is not None:
        #         pickup_advantage = best_busy_pickup - cand.pickup_eta_new
        #         if pickup_advantage >= 20.0:
        #             scores[i] += min(24.0, 0.40 * pickup_advantage)
        #         elif pickup_advantage > 0:
        #             scores[i] += min(10.0, 0.18 * pickup_advantage)
        #     if (not is_idle) and best_idle_pickup is not None:
        #         slower_than_idle = cand.pickup_eta_new - best_idle_pickup
        #         if slower_than_idle >= 20.0:
        #             scores[i] -= min(22.0, 0.30 * slower_than_idle)
        #     # Near max-wait violations should strongly prefer earlier pickup.
        #     slack_after = getattr(request, "max_wait", 300.0) - max(0.0, cand.pickup_eta_new - request.request_time)
        #     if slack_after < 45.0:
        #         scores[i] += max(0.0, (45.0 - slack_after) * 0.35)

        _print_top5(candidates, scores, request, self.taxi_plans, now)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = candidates[best_idx]

        if best.is_defer:
            request.status = RequestStatus.DEFERRED
            _log(f"  → DEFERRED request {request.request_id}")
            return best

        # Record the assignment in our plan — plan.stops is updated so that
        # subsequent requests in the same tick see the updated stop list and
        # generate correct insertion candidates for the same taxi.
        plan = self.taxi_plans[best.taxi_id]

        # Take a rollback snapshot before the FIRST mutation for this taxi in
        # the current tick. This preserves the previously committed SUMO suffix
        # so a failed re-dispatch does not wipe onboard / already-assigned work.
        if best.taxi_id not in self._dispatch_snapshots:
            self._dispatch_snapshots[best.taxi_id] = (
                _clone_stops(plan.stops),
                set(plan.assigned_request_ids),
            )

        plan.stops = best.resulting_stops
        # NOTE: this set is currently person-id keyed elsewhere in the file.
        # Keep that convention here to avoid broad behavioral changes in this patch.
        plan.assigned_request_ids.add(request.person_id)
        request.assigned_taxi_id = best.taxi_id
        request.status = RequestStatus.ASSIGNED

        # Mark this taxi as needing a SUMO dispatch call this tick.
        # _flush_idle_dispatches will issue the single dispatchTaxi call after
        # all requests have been processed.
        self._pending_dispatches.add(best.taxi_id)

        _log(f"  → ASSIGNED req={request.request_id} → taxi={best.taxi_id}"
             f"  pu_eta={best.pickup_eta_new:.1f}s  (flush pending)")

        return best

    def _flush_idle_dispatches(self) -> None:
        """
        For each taxi that received assignments this tick, issue exactly ONE
        dispatchTaxi call with the complete ordered reservation list.

        Critical SUMO rule:
        - the list must encode the FULL REMAINING reservation chain
        - not just upcoming pickups
        - onboard customers therefore remain represented by their remaining
          DROPOFF stop(s) in plan.stops
        """
        for taxi_id in list(self._pending_dispatches):
            self._pending_dispatches.discard(taxi_id)
            plan = self.taxi_plans.get(taxi_id)
            if plan is None:
                self._dispatch_snapshots.pop(taxi_id, None)
                continue

            ordered_res_ids = _serialize_dispatch_res_ids(plan)

            if not ordered_res_ids:
                self._dispatch_snapshots.pop(taxi_id, None)
                continue

            try:
                traci.vehicle.dispatchTaxi(taxi_id, ordered_res_ids)
                _log(f"  → DISPATCHED taxi={taxi_id} reservations={ordered_res_ids}")
                self._dispatch_snapshots.pop(taxi_id, None)
            except traci.TraCIException as e:
                _log(f"  [ERROR] dispatchTaxi failed for taxi={taxi_id}: {e}")
                _log(f"          ordered_res_ids = {ordered_res_ids}")

                prev_stops, prev_assigned_ids = self._dispatch_snapshots.pop(
                    taxi_id, (_clone_stops(plan.stops), set(plan.assigned_request_ids))
                )
                new_assigned_ids = set(plan.assigned_request_ids) - set(prev_assigned_ids)

                # Roll back ONLY the tentative assignments introduced this tick.
                for pid in new_assigned_ids:
                    req = self.requests.get(pid)
                    if req and req.status == RequestStatus.ASSIGNED and req.assigned_taxi_id == taxi_id:
                        req.assigned_taxi_id = None
                        req.status = RequestStatus.PENDING

                plan.stops = prev_stops
                plan.assigned_request_ids = set(prev_assigned_ids)


    def _termination_ready(self) -> bool:
        """Return True only when there is no tracked or SUMO-visible work left."""
        try:
            active_persons = set(traci.person.getIDList())
        except traci.TraCIException:
            active_persons = set()

        try:
            active_vids = set(traci.vehicle.getIDList())
        except traci.TraCIException:
            active_vids = set()

        sumo_onboard = False
        for vid in active_vids:
            try:
                if traci.vehicle.getPersonIDList(vid):
                    sumo_onboard = True
                    break
            except traci.TraCIException:
                continue

        tracked_open = any(r.status != RequestStatus.COMPLETED for r in self.requests.values())
        local_taxi_work = any(
            plan.stops or plan.assigned_request_ids or plan.onboard_count > 0
            for plan in self.taxi_plans.values()
        )

        try:
            min_expected = traci.simulation.getMinExpectedNumber()
        except traci.TraCIException:
            min_expected = 1

        return (
            min_expected == 0
            and not active_persons
            and not sumo_onboard
            and not tracked_open
            and not local_taxi_work
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        _log("\n" + "=" * 72)
        _log("  SUMO Heuristic DRT Dispatcher starting...")
        _log("=" * 72)

        while True:
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                _log("  [INFO] TraCI connection closed by SUMO during step — ending simulation loop.")
                break

            try:
                now = traci.simulation.getTime()
            except traci.exceptions.FatalTraCIError:
                _log("  [INFO] TraCI connection closed by SUMO — ending simulation loop.")
                break

            self._step_count += 1

            # --- per-step accumulator update ---
            dt = self.step_length
            pending_count = sum(
                1 for r in self.requests.values()
                if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)
            )
            self.accumulator.wait_cost += pending_count * dt
            self.accumulator.elapsed_time += dt

            # empty driving cost
            _refresh_taxi_plans(self.taxi_plans)
            for plan in self.taxi_plans.values():
                try:
                    dist = traci.vehicle.getDistance(plan.taxi_id)
                    delta = dist - plan.cumulative_distance
                    plan.cumulative_distance = dist
                    if plan.onboard_count == 0 and delta > 0:
                        self.accumulator.empty_dist_cost += delta
                except traci.TraCIException:
                    pass

            # --- tick boundary ---
            if self._step_count >= self.TICK_STEPS:
                self._step_count = 0
                self._tick_num += 1
                self._process_tick(now)

            # Robust termination check must happen AFTER stepping and after any
            # tick processing, otherwise we may stop one step too early while a
            # passenger is still onboard or a final dropoff event has not yet
            # been synchronised into local request state.
            try:
                if self._termination_ready():
                    _log("  [INFO] All tracked requests completed and no active work remains — simulation complete.")
                    break
            except traci.exceptions.FatalTraCIError:
                _log("  [INFO] TraCI connection closed by SUMO — ending simulation loop.")
                break

        _log("\n" + "=" * 72)
        _log("  Simulation complete.")
        self._print_final_summary()
        _log("=" * 72)

    def _process_tick(self, now: float) -> None:
        """Called every 10 steps. Syncs state, detects events, dispatches."""
        # remove taxis that have left the simulation so no API call is made on them
        """ The following might lines might be useful in future if there is problem of
        dispatching unknown vehicles"""
        # active_vids = set(traci.vehicle.getIDList())
        # _log(f"Active VIDs: {active_vids}")
        # departed = [tid for tid in self.taxi_plans if tid not in active_vids]
        # if departed:
        #     _log(f"Departed: {departed}")
        # for tid in departed:
        #     _log(f"  [INFO] taxi {tid} left simulation — removing from fleet")
        #     del self.taxi_plans[tid]

        # sync reservations and status
        self._sync_reservations(now)
        _sync_onboard(self.taxi_plans, self.requests)

        # detect meaningful events
        had_event, new_arrivals, new_pickups, new_dropoffs = _detect_events(
            self._prev_req_ids,
            self._prev_onboard_ids,
            self._prev_completed_ids,
            self.requests,
        )

        # update dropoff accumulator
        self.accumulator.completed_dropoffs += len(new_dropoffs)
        for pid in new_dropoffs:
            req = self.requests.get(pid)
            if req and req.excess_ride_time is not None:
                self.accumulator.ride_cost += req.excess_ride_time

        # build tick context using the periodic-dispatch rule:
        # a tick is only meaningful if there is a pending pool AND at least one
        # feasible candidate on a taxi that is safe to replan this tick.
        pending_pool = [pid for pid, r in self.requests.items()
                        if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
        self._eligible_taxis_this_tick = _eligible_taxis_for_tick(
            self.taxi_plans, self.requests, new_pickups, new_dropoffs,
        )
        request_lookup = _build_request_lookup_by_res_id(self.requests)
        
        # _log(f"PENDING POOL: {pending_pool}")

        has_candidates = False
        if pending_pool and self._eligible_taxis_this_tick:
            for pid in sorted(pending_pool, key=lambda rid: self.requests[rid].waiting_time(now), reverse=True):
                req = self.requests[pid]
                cands = generate_candidates(
                    req,
                    self.taxi_plans,
                    self.requests,
                    now,
                    eligible_taxi_ids=self._eligible_taxis_this_tick,
                    request_lookup_by_res_id=request_lookup,
                )
                if any(not c.is_defer for c in cands):
                    has_candidates = True
                    break

        outcome = TickOutcome.MEANINGFUL if has_candidates else TickOutcome.IDLE

        tick = TickContext(
            outcome=outcome,
            pending_pool=pending_pool,
            has_candidates=has_candidates,
            sim_time=now,
        )

        # print tick summary on event ticks or any tick that exposes actionable candidates
        if had_event or has_candidates:
            _print_tick_summary(
                self._tick_num, now, tick,
                new_arrivals, new_pickups, new_dropoffs,
                self.requests, self.taxi_plans,
                self.accumulator,
            )
            _log(f"  Replanable taxis this tick: {sorted(self._eligible_taxis_this_tick)}")
            _log("  Future taxi stop plans:")
            for taxi_id, plan in sorted(self.taxi_plans.items()):
                stops = plan.stops
                if not stops:
                    _log(f"    {taxi_id}: []")
                    continue

                stop_list = [
                    f"{'PU' if s.stop_type == StopType.PICKUP else 'DO'}({s.request_id})@{s.eta:.1f}"
                    for s in stops
                ]
                _log(f"    {taxi_id}: [{', '.join(stop_list)}]")

        # dispatch pending requests if meaningful
        if outcome == TickOutcome.MEANINGFUL:
            sorted_pool = sorted(
                pending_pool,
                key=lambda pid: self.requests[pid].waiting_time(now),
                reverse=True,
            )
            for pid in sorted_pool:
                req = self.requests.get(pid)
                if req and req.status in (RequestStatus.PENDING, RequestStatus.DEFERRED):
                    self._dispatch_best(req, now)

            # Issue one dispatchTaxi per taxi for all assignments made this tick
            self._flush_idle_dispatches()

            # reset accumulator after acting
            self.accumulator.reset()

        # update snapshots for next tick
        self._prev_req_ids       = set(self.requests.keys())
        self._prev_onboard_ids   = {pid for pid, r in self.requests.items()
                                    if r.status == RequestStatus.ONBOARD}
        self._prev_completed_ids = {pid for pid, r in self.requests.items()
                                    if r.status == RequestStatus.COMPLETED}

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def _print_final_summary(self) -> None:
        completed = [r for r in self.requests.values() if r.status == RequestStatus.COMPLETED]
        onboard = [r for r in self.requests.values() if r.status == RequestStatus.ONBOARD]
        assigned = [r for r in self.requests.values() if r.status == RequestStatus.ASSIGNED]
        pending = [r for r in self.requests.values()
                   if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
        unfinished = [r for r in self.requests.values() if r.status != RequestStatus.COMPLETED]

        wait_times = [
            (r.pickup_time - r.request_time)
            for r in completed if r.pickup_time is not None
        ]
        excess_rides = [r.excess_ride_time for r in completed if r.excess_ride_time is not None]

        _log(f"\n  Total requests   : {len(self.requests)}")
        _log(f"  Completed        : {len(completed)}")
        _log(f"  Pending/Deferred : {len(pending)}")
        _log(f"  Assigned         : {len(assigned)}")
        _log(f"  Onboard          : {len(onboard)}")
        _log(f"  Not completed    : {len(unfinished)}")
        if wait_times:
            _log(f"  Avg pickup wait  : {sum(wait_times)/len(wait_times):.1f}s"
                  f"  max={max(wait_times):.1f}s")
        if excess_rides:
            _log(f"  Avg excess ride  : {sum(excess_rides)/len(excess_rides):.1f}s"
                  f"  max={max(excess_rides):.1f}s")
        normalizer_rows = _SCORE_NORMALIZER.get_summary_rows()
        if normalizer_rows:
            _log("\n  Learned score-normalizer statistics:")
            for row in normalizer_rows:
                _log(
                    f"    {row['metric']:<24} "
                    f"count={row['count']:>5}  "
                    f"mean={row['mean']:>10.4f}  "
                    f"std={row['std']:>10.4f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic DRT dispatcher")
    parser.add_argument("--cfg",         required=True,       help="Path to .sumocfg")
    parser.add_argument("--gui",         action="store_true", help="Use sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--log-file",    type=str,   default=None,
                        help="Log file path. Defaults to dispatcher_YYYYMMDD_HHMMSS.log")
    args = parser.parse_args()

    # --- set up logger ---
    global log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = args.log_file or f"dispatcher_{timestamp}.log"
    log = setup_logger(log_path)
    _log(f"  Log file: {log_path}")

    dispatcher = HeuristicDispatcher(
        cfg_path=args.cfg,
        step_length=args.step_length,
        use_gui=args.gui,
    )
    dispatcher.start()
    try:
        dispatcher.run()
    except KeyboardInterrupt:
        _log("\n  [INFO] Interrupted by user.")
    finally:
        dispatcher.close()
        _log(f"\n  Full log saved to: {log_path}")


if __name__ == "__main__":
    main()