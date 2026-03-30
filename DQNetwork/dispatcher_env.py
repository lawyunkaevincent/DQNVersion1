from __future__ import annotations

from typing import Optional

import traci

from DRTDataclass import (
    CandidateInsertion,
    GlobalStateSummary,
    Request,
    RequestStatus,
    TaxiStatus,
    TickContext,
    TickOutcome,
)
from dispatcher import (
    HeuristicDispatcher as LegacyHeuristicDispatcher,
    _build_request_lookup_by_res_id,
    _clone_stops,
    _detect_events,
    _eligible_taxis_for_tick,
    _log,
    _print_tick_summary,
    _print_top5,
    _serialize_dispatch_res_ids,
    _sync_onboard,
    generate_candidates,
)
from drt_policy_types import DecisionPoint, PolicyOutput
from heuristic_policy import BasePolicy, HeuristicPolicy
from dataset_logger import ImitationDatasetLogger


class RefactoredDRTEnvironment(LegacyHeuristicDispatcher):
    """
    Agent-driven wrapper around the original heuristic dispatcher.

    This version intentionally follows dispatcher.py as closely as possible:
      - no speculative plan pruning
      - request/taxi synchronization stays event-driven
      - apply_action mutates plan.stops exactly once per chosen action
      - _flush_idle_dispatches serializes the full current suffix and dispatches it
      - rollback restores the pre-tick snapshot on dispatch failure

    The only added behavior is exposing policy-facing decision helpers.
    """

    def __init__(
        self,
        cfg_path: str,
        step_length: float = 1.0,
        use_gui: bool = False,
        policy: Optional[BasePolicy] = None,
        dataset_logger: Optional[ImitationDatasetLogger] = None,
    ):
        super().__init__(cfg_path=cfg_path, step_length=step_length, use_gui=use_gui)
        self.policy: BasePolicy = policy or HeuristicPolicy()
        self.dataset_logger = dataset_logger
        self._decision_counter = 0

    # ------------------------------------------------------------------
    # Synchronization: keep legacy event-driven behavior
    # ------------------------------------------------------------------

    def _sync_reservations(self, now: float) -> None:
        super()._sync_reservations(now)

        # Legacy-compatible cleanup: only clear completed requests from local taxi plans.
        # Track which taxis had stops removed so we can re-dispatch them.
        taxis_needing_redispatch: set[str] = set()
        for req in self.requests.values():
            if req.status == RequestStatus.COMPLETED and req.assigned_taxi_id in self.taxi_plans:
                plan = self.taxi_plans[req.assigned_taxi_id]
                old_len = len(plan.stops)
                plan.stops = [s for s in plan.stops if s.request_id != req.request_id]
                if len(plan.stops) != old_len:
                    taxis_needing_redispatch.add(req.assigned_taxi_id)
                plan.assigned_request_ids.discard(req.person_id)
                plan.assigned_request_ids.discard(req.request_id)
                plan.onboard_request_ids.discard(req.person_id)
                plan.onboard_request_ids.discard(req.request_id)
                plan.onboard_count = len(plan.onboard_request_ids)

        # Re-dispatch taxis that had stops removed so SUMO continues routing
        for taxi_id in taxis_needing_redispatch:
            plan = self.taxi_plans.get(taxi_id)
            if plan and plan.stops:
                self._redispatch_remaining(taxi_id, plan)

    # ------------------------------------------------------------------
    # New refactored decision helpers
    # ------------------------------------------------------------------

    def build_global_state_summary(self, now: float) -> GlobalStateSummary:
        pending = [r for r in self.requests.values() if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
        onboard_count = sum(plan.onboard_count for plan in self.taxi_plans.values())
        idle_taxi_count = sum(
            1 for plan in self.taxi_plans.values()
            if plan.status == TaxiStatus.IDLE and plan.onboard_count == 0
        )
        active_taxi_count = sum(
            1 for plan in self.taxi_plans.values()
            if not (plan.status == TaxiStatus.IDLE and len(plan.stops) == 0 and plan.onboard_count == 0)
        )
        avg_wait = (sum(r.waiting_time(now) for r in pending) / len(pending)) if pending else 0.0
        max_wait = max((r.waiting_time(now) for r in pending), default=0.0)

        taxi_count = max(1, len(self.taxi_plans))
        avg_occupancy = sum((plan.onboard_count / max(1, plan.capacity)) for plan in self.taxi_plans.values()) / taxi_count
        fleet_utilization = sum(
            1 for plan in self.taxi_plans.values()
            if plan.status != TaxiStatus.IDLE or plan.onboard_count > 0
        ) / taxi_count

        recent_window = 60.0
        recent_count = sum(1 for r in self.requests.values() if 0.0 <= now - r.request_time <= recent_window)
        recent_demand_rate = recent_count / recent_window

        return GlobalStateSummary(
            sim_time=now,
            pending_req_count=len(pending),
            onboard_count=onboard_count,
            idle_taxi_count=idle_taxi_count,
            active_taxi_count=active_taxi_count,
            avg_wait_time=avg_wait,
            max_wait_time=max_wait,
            avg_occupancy=avg_occupancy,
            fleet_utilization=fleet_utilization,
            recent_demand_rate=recent_demand_rate,
        )

    def build_candidates_for_request(self, request: Request, now: float) -> list[CandidateInsertion]:
        request_lookup = _build_request_lookup_by_res_id(self.requests)
        eligible_taxis = getattr(self, "_eligible_taxis_this_tick", set())
        return generate_candidates(
            request,
            self.taxi_plans,
            self.requests,
            now,
            eligible_taxi_ids=eligible_taxis,
            request_lookup_by_res_id=request_lookup,
        )

    def build_decision_point(
        self,
        request: Request,
        now: float,
        tick_context: Optional[TickContext] = None,
    ) -> DecisionPoint | None:
        candidates = self.build_candidates_for_request(request, now)
        if not candidates:
            return None
        self._decision_counter += 1
        decision_id = f"tick{self._tick_num:05d}_d{self._decision_counter:06d}_req{request.request_id}"
        return DecisionPoint(
            request=request,
            state_summary=self.build_global_state_summary(now),
            candidate_actions=candidates,
            sim_time=now,
            tick_context=tick_context,
            decision_id=decision_id,
        )

    def apply_action(self, request: Request, action: CandidateInsertion, now: float) -> CandidateInsertion:
        if action.is_defer:
            request.status = RequestStatus.DEFERRED
            _log(f"  → DEFERRED request {request.request_id}")
            return action

        plan = self.taxi_plans[action.taxi_id]
        if action.taxi_id not in self._dispatch_snapshots:
            self._dispatch_snapshots[action.taxi_id] = (
                _clone_stops(plan.stops),
                set(plan.assigned_request_ids),
            )

        # Keep the legacy convention: the chosen candidate replaces the taxi's
        # future stop suffix, and assigned_request_ids remains person-id keyed.
        plan.stops = list(action.resulting_stops)
        plan.assigned_request_ids.add(request.person_id)

        request.assigned_taxi_id = action.taxi_id
        request.status = RequestStatus.ASSIGNED
        self._pending_dispatches.add(action.taxi_id)

        _log(
            f"  → ASSIGNED req={request.request_id} → taxi={action.taxi_id} "
            f" pu_eta={action.pickup_eta_new:.1f}s  (flush pending)"
        )
        return action

    def dispatch_request_via_policy(
        self,
        request: Request,
        now: float,
        tick_context: Optional[TickContext] = None,
    ) -> PolicyOutput | None:
        decision = self.build_decision_point(request, now, tick_context=tick_context)
        if decision is None:
            return None

        policy_output = self.policy.select_action(decision, self.taxi_plans, now)
        if getattr(self.policy, "print_top_k", False):
            candidates = [e.candidate for e in policy_output.evaluations]
            scores = [e.score for e in policy_output.evaluations]
            _print_top5(candidates, scores, request, self.taxi_plans, now)

        self.apply_action(request, policy_output.chosen_action, now)

        if self.dataset_logger is not None:
            self.dataset_logger.log_decision(decision, policy_output, self.taxi_plans)

        return policy_output

    # ------------------------------------------------------------------
    # Dispatch override: mirror legacy batching / rollback behavior
    # ------------------------------------------------------------------

    # def _flush_idle_dispatches(self) -> None:
    #     if not self._pending_dispatches:
    #         return

    #     try:
    #         active_vids = set(traci.vehicle.getIDList())
    #     except Exception:
    #         active_vids = set()

    #     for taxi_id in list(self._pending_dispatches):
    #         self._pending_dispatches.discard(taxi_id)

    #         if taxi_id not in active_vids:
    #             self._dispatch_snapshots.pop(taxi_id, None)
    #             continue

    #         plan = self.taxi_plans.get(taxi_id)
    #         if plan is None:
    #             self._dispatch_snapshots.pop(taxi_id, None)
    #             continue

    #         # --- Scrub stale reservation IDs before dispatching ---
    #         # SUMO can silently close a reservation (e.g. when a taxi misses a
    #         # pickup window mid-route) without informing TraCI. This causes two
    #         # distinct errors depending on whether the passenger is onboard:
    #         #
    #         #   "Reservation id 'X' is not known"
    #         #     → passenger was never picked up; SUMO discarded the reservation.
    #         #     → safe to remove ALL stops for this request from the plan.
    #         #
    #         #   "Re-dispatch did mention some customers but failed to mention X"
    #         #     → passenger IS physically onboard the taxi right now; SUMO
    #         #       still tracks them via the vehicle's passenger list even though
    #         #       the reservation object is gone. Every dispatchTaxi call MUST
    #         #       include their dropoff stop or SUMO rejects the call.
    #         #     → keep the DROPOFF stop; only remove the (already consumed)
    #         #       PICKUP stop.
    #         #
    #         # We therefore query both the live reservation set AND the set of
    #         # persons currently inside the taxi to distinguish these two cases.
    #         try:
    #             live_res_ids = {r.id for r in traci.person.getTaxiReservations(0)}
    #         except Exception:
    #             live_res_ids = None  # can't verify — proceed as-is

    #         try:
    #             persons_in_taxi = set(traci.vehicle.getPersonIDList(taxi_id))
    #         except Exception:
    #             persons_in_taxi = set()

    #         if live_res_ids is not None:
    #             stale_rids = {s.request_id for s in plan.stops
    #                           if s.request_id not in live_res_ids}
    #             if stale_rids:
    #                 _log(f"  [WARN] stale reservation IDs for taxi={taxi_id}: {stale_rids}")
    #                 for stale_rid in stale_rids:
    #                     pid = self.resid_to_pid.get(stale_rid, stale_rid)
    #                     req = self.requests.get(pid)
    #                     is_onboard = pid in persons_in_taxi

    #                     if is_onboard:
    #                         # Passenger is physically inside the taxi but their
    #                         # reservation object is already closed in SUMO.
    #                         # SUMO will complete the dropoff autonomously.
    #                         # Including this ID in dispatchTaxi causes "not known"
    #                         # errors; omitting an onboard passenger causes "failed
    #                         # to mention" errors. The only safe path: remove ALL
    #                         # stops for this request and let SUMO finish the ride
    #                         # on its own. Mark the request ONBOARD so our state
    #                         # stays consistent until _sync_reservations detects
    #                         # the dropoff (person leaves sim).
    #                         plan.stops = [s for s in plan.stops
    #                                       if s.request_id != stale_rid]
    #                         plan.assigned_request_ids.discard(stale_rid)
    #                         plan.assigned_request_ids.discard(pid)
    #                         plan.onboard_request_ids.add(pid)
    #                         if req and req.status != RequestStatus.ONBOARD:
    #                             req.status = RequestStatus.ONBOARD
    #                             if req.pickup_time is None:
    #                                 req.pickup_time = traci.simulation.getTime()
    #                         _log(f"    → {stale_rid} onboard with closed reservation — removed stops, SUMO handles dropoff autonomously")
    #                     else:
    #                         # Passenger was never picked up — SUMO discarded the
    #                         # reservation entirely. Safe to remove all stops and
    #                         # mark the request completed so it won't re-queue.
    #                         plan.stops = [s for s in plan.stops
    #                                       if s.request_id != stale_rid]
    #                         plan.assigned_request_ids.discard(stale_rid)
    #                         plan.assigned_request_ids.discard(pid)
    #                         plan.onboard_request_ids.discard(stale_rid)
    #                         plan.onboard_request_ids.discard(pid)
    #                         if req and req.status not in (RequestStatus.COMPLETED,):
    #                             req.status = RequestStatus.COMPLETED
    #                             _log(f"    → {stale_rid} never picked up — removed all stops, marked COMPLETED")
    #                 plan.onboard_count = len(plan.onboard_request_ids)

    #         ordered_res_ids = _serialize_dispatch_res_ids(plan)
    #         if not ordered_res_ids:
    #             self._dispatch_snapshots.pop(taxi_id, None)
    #             continue

    #         try:
    #             traci.vehicle.dispatchTaxi(taxi_id, ordered_res_ids)
    #             _log(f"  → DISPATCHED taxi={taxi_id} reservations={ordered_res_ids}")
    #             self._dispatch_snapshots.pop(taxi_id, None)
    #         except traci.TraCIException as e:
    #             _log(f" ")
    #             _log(f"  [ERROR] dispatchTaxi failed for taxi={taxi_id}: {e}")
    #             _log(f" Current passenger onboard: {traci.vehicle.getPersonIDList(taxi_id)}")
    #             _log(f"          ordered_res_ids = {ordered_res_ids}")

    #             prev_stops, prev_assigned_ids = self._dispatch_snapshots.pop(
    #                 taxi_id, (_clone_stops(plan.stops), set(plan.assigned_request_ids))
    #             )
    #             new_assigned_ids = set(plan.assigned_request_ids) - set(prev_assigned_ids)

    #             # Roll back only tentative assignments introduced this tick.
    #             for pid in new_assigned_ids:
    #                 req = self.requests.get(pid)
    #                 if req and req.status == RequestStatus.ASSIGNED and req.assigned_taxi_id == taxi_id:
    #                     req.assigned_taxi_id = None
    #                     req.status = RequestStatus.PENDING

    #             plan.stops = prev_stops
    #             plan.assigned_request_ids = set(prev_assigned_ids)

    def _flush_idle_dispatches(self) -> None:
        if not self._pending_dispatches:
            return

        try:
            active_vids = set(traci.vehicle.getIDList())
        except Exception:
            active_vids = set()

        for taxi_id in list(self._pending_dispatches):
            self._pending_dispatches.discard(taxi_id)

            if taxi_id not in active_vids:
                self._dispatch_snapshots.pop(taxi_id, None)
                continue

            plan = self.taxi_plans.get(taxi_id)
            if plan is None:
                self._dispatch_snapshots.pop(taxi_id, None)
                continue

            # --- Scrub stale reservation IDs before dispatching ---
            try:
                live_res_ids = {r.id for r in traci.person.getTaxiReservations(0)}
            except Exception:
                live_res_ids = None  # can't verify — proceed as-is

            try:
                persons_in_taxi = set(traci.vehicle.getPersonIDList(taxi_id))
            except Exception:
                persons_in_taxi = set()

            # NEW: use actual person presence in the sim as the source of truth
            try:
                active_pids = set(traci.person.getIDList())
            except Exception:
                active_pids = set()

            if live_res_ids is not None:
                stale_rids = {
                    s.request_id for s in plan.stops
                    if s.request_id not in live_res_ids
                }

                if stale_rids:
                    _log(f"  [WARN] stale reservation IDs for taxi={taxi_id}: {stale_rids}")

                    for stale_rid in stale_rids:
                        pid = self.resid_to_pid.get(stale_rid, stale_rid)
                        req = self.requests.get(pid)
                        is_onboard = pid in persons_in_taxi
                        in_sim = pid in active_pids

                        if is_onboard:
                            # Passenger is physically inside the taxi, but the
                            # reservation object has already closed in SUMO.
                            # Remove local stops for this request and let SUMO
                            # finish the dropoff autonomously.
                            plan.stops = [
                                s for s in plan.stops
                                if s.request_id != stale_rid
                            ]
                            plan.assigned_request_ids.discard(stale_rid)
                            plan.assigned_request_ids.discard(pid)
                            plan.onboard_request_ids.add(pid)

                            if req and req.status != RequestStatus.ONBOARD:
                                req.status = RequestStatus.ONBOARD
                                req.assigned_taxi_id = taxi_id
                                if req.pickup_time is None:
                                    req.pickup_time = traci.simulation.getTime()

                            _log(
                                f"    → {stale_rid} onboard with closed reservation — "
                                f"removed stops, SUMO handles dropoff autonomously"
                            )

                        elif not in_sim:
                            # Person has actually left the simulation.
                            # This is the only safe non-onboard case to purge and complete.
                            plan.stops = [
                                s for s in plan.stops
                                if s.request_id != stale_rid
                            ]
                            plan.assigned_request_ids.discard(stale_rid)
                            plan.assigned_request_ids.discard(pid)
                            plan.onboard_request_ids.discard(stale_rid)
                            plan.onboard_request_ids.discard(pid)

                            if req and req.status != RequestStatus.COMPLETED:
                                req.status = RequestStatus.COMPLETED

                            _log(
                                f"    → {stale_rid} person no longer in sim — "
                                f"removed stops, marked COMPLETED"
                            )

                        else:
                            # Reservation is dead in SUMO but person is still in
                            # sim waiting. SUMO has unilaterally cancelled this
                            # reservation (e.g. after the taxi completed an earlier
                            # passenger's service and went idle before reaching
                            # this pickup). We MUST remove these stops or the taxi
                            # will be permanently broken — every future dispatchTaxi
                            # call that includes this dead ID will fail.
                            #
                            # The person may still get a fresh reservation from
                            # SUMO on the next _sync_reservations tick. Reset the
                            # request to PENDING so it re-enters the dispatch pool.
                            plan.stops = [
                                s for s in plan.stops
                                if s.request_id != stale_rid
                            ]
                            plan.assigned_request_ids.discard(stale_rid)
                            plan.assigned_request_ids.discard(pid)
                            plan.onboard_request_ids.discard(stale_rid)
                            plan.onboard_request_ids.discard(pid)

                            if req is not None:
                                req.assigned_taxi_id = None
                                req.status = RequestStatus.PENDING

                            _log(
                                f"    → {stale_rid} reservation dead in SUMO, person still "
                                f"waiting — purged stops, reset to PENDING for re-dispatch"
                            )

                    plan.onboard_count = len(plan.onboard_request_ids)

            ordered_res_ids = _serialize_dispatch_res_ids(plan)

            if not ordered_res_ids:
                self._dispatch_snapshots.pop(taxi_id, None)
                continue

            try:
                traci.vehicle.dispatchTaxi(taxi_id, ordered_res_ids)
                _log(f"  → DISPATCHED taxi={taxi_id} reservations={ordered_res_ids}")
                self._dispatch_snapshots.pop(taxi_id, None)

            except traci.TraCIException as e:
                _log(" ")
                _log(f"  [ERROR] dispatchTaxi failed for taxi={taxi_id}: {e}")
                _log(f" Current passenger onboard: {traci.vehicle.getPersonIDList(taxi_id)}")
                _log(f"          ordered_res_ids = {ordered_res_ids}")

                # ── Attempt to retry after purging the dead reservation ──
                import re
                match = re.search(r"Reservation id '(\w+)' is not known", str(e))
                if match:
                    dead_rid = match.group(1)
                    _log(f"  [RETRY] purging dead reservation '{dead_rid}' and retrying dispatch")
                    plan.stops = [s for s in plan.stops if s.request_id != dead_rid]
                    dead_pid = self.resid_to_pid.get(dead_rid, dead_rid)
                    plan.assigned_request_ids.discard(dead_rid)
                    plan.assigned_request_ids.discard(dead_pid)
                    plan.onboard_request_ids.discard(dead_rid)
                    plan.onboard_request_ids.discard(dead_pid)
                    dead_req = self.requests.get(dead_pid)
                    if dead_req and dead_req.status not in (RequestStatus.ONBOARD, RequestStatus.COMPLETED):
                        dead_req.assigned_taxi_id = None
                        dead_req.status = RequestStatus.PENDING

                    retry_res_ids = _serialize_dispatch_res_ids(plan)
                    if retry_res_ids:
                        try:
                            traci.vehicle.dispatchTaxi(taxi_id, retry_res_ids)
                            _log(f"  → RETRY DISPATCHED taxi={taxi_id} reservations={retry_res_ids}")
                            self._dispatch_snapshots.pop(taxi_id, None)
                            continue  # success — skip rollback
                        except traci.TraCIException as retry_e:
                            _log(f"  [RETRY-FAIL] {retry_e}")
                    elif not plan.stops:
                        # All stops were dead — nothing to dispatch
                        self._dispatch_snapshots.pop(taxi_id, None)
                        continue

                # ── Full rollback if retry didn't work ──
                prev_stops, prev_assigned_ids = self._dispatch_snapshots.pop(
                    taxi_id, (_clone_stops(plan.stops), set(plan.assigned_request_ids))
                )
                new_assigned_ids = set(plan.assigned_request_ids) - set(prev_assigned_ids)

                # Roll back only tentative assignments introduced this tick.
                for pid in new_assigned_ids:
                    req = self.requests.get(pid)
                    if req and req.status == RequestStatus.ASSIGNED and req.assigned_taxi_id == taxi_id:
                        req.assigned_taxi_id = None
                        req.status = RequestStatus.PENDING

                plan.stops = prev_stops
                plan.assigned_request_ids = set(prev_assigned_ids)

                # ── Post-rollback: purge dead reservations from restored stops ──
                # The rolled-back snapshot may itself contain dead reservation IDs
                # (e.g. 142 in the original bug). Without this check the dead ID
                # persists forever and blocks every future dispatch for this taxi.
                try:
                    post_live = {r.id for r in traci.person.getTaxiReservations(0)}
                except Exception:
                    post_live = None

                if post_live is not None:
                    dead_in_restored = {
                        s.request_id for s in plan.stops
                        if s.request_id not in post_live
                    }
                    if dead_in_restored:
                        _log(f"  [ROLLBACK-CLEANUP] purging dead IDs from restored plan: {dead_in_restored}")
                        for dead_rid in dead_in_restored:
                            plan.stops = [s for s in plan.stops if s.request_id != dead_rid]
                            dead_pid = self.resid_to_pid.get(dead_rid, dead_rid)
                            plan.assigned_request_ids.discard(dead_rid)
                            plan.assigned_request_ids.discard(dead_pid)
                            plan.onboard_request_ids.discard(dead_rid)
                            plan.onboard_request_ids.discard(dead_pid)
                            dead_req = self.requests.get(dead_pid)
                            if dead_req and dead_req.status not in (RequestStatus.ONBOARD, RequestStatus.COMPLETED):
                                dead_req.assigned_taxi_id = None
                                dead_req.status = RequestStatus.PENDING
                        plan.onboard_count = len(plan.onboard_request_ids)

    # ------------------------------------------------------------------
    # Refactored tick loop: mechanics separated from policy
    # ------------------------------------------------------------------

    def _process_tick(self, now: float) -> None:
        self._sync_reservations(now)
        _sync_onboard(self.taxi_plans, self.requests)

        had_event, new_arrivals, new_pickups, new_dropoffs = _detect_events(
            self._prev_req_ids,
            self._prev_onboard_ids,
            self._prev_completed_ids,
            self.requests,
        )

        self.accumulator.completed_dropoffs += len(new_dropoffs)
        for pid in new_dropoffs:
            req = self.requests.get(pid)
            if req and req.excess_ride_time is not None:
                self.accumulator.ride_cost += req.excess_ride_time

        pending_pool = [pid for pid, r in self.requests.items() if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
        self._eligible_taxis_this_tick = _eligible_taxis_for_tick(
            self.taxi_plans,
            self.requests,
            new_pickups,
            new_dropoffs,
        )

        has_candidates = False
        if pending_pool and self._eligible_taxis_this_tick:
            for pid in sorted(pending_pool, key=lambda rid: self.requests[rid].waiting_time(now), reverse=True):
                req = self.requests[pid]
                cands = self.build_candidates_for_request(req, now)
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

        if had_event or has_candidates:
            _print_tick_summary(
                self._tick_num,
                now,
                tick,
                new_arrivals,
                new_pickups,
                new_dropoffs,
                self.requests,
                self.taxi_plans,
                self.accumulator,
            )
            _log(f"  Replanable taxis this tick: {sorted(self._eligible_taxis_this_tick)}")
            _log("  Future taxi stop plans:")
            for taxi_id, plan in sorted(self.taxi_plans.items()):
                if not plan.stops:
                    _log(f"    {taxi_id}: []")
                    continue
                stop_list = [
                    f"{'PU' if s.stop_type.name == 'PICKUP' else 'DO'}({s.request_id})@{s.eta:.1f}"
                    for s in plan.stops
                ]
                _log(f"    {taxi_id}: [{', '.join(stop_list)}]")

        if outcome == TickOutcome.MEANINGFUL:
            sorted_pool = sorted(
                pending_pool,
                key=lambda pid: self.requests[pid].waiting_time(now),
                reverse=True,
            )
            for pid in sorted_pool:
                req = self.requests.get(pid)
                if req and req.status in (RequestStatus.PENDING, RequestStatus.DEFERRED):
                    self.dispatch_request_via_policy(req, now, tick_context=tick)

            self._flush_idle_dispatches()
            self.accumulator.reset()

        self._prev_req_ids = set(self.requests.keys())
        self._prev_onboard_ids = {pid for pid, r in self.requests.items() if r.status == RequestStatus.ONBOARD}
        self._prev_completed_ids = {pid for pid, r in self.requests.items() if r.status == RequestStatus.COMPLETED}

    def _debug_check_plan_consistency(self, tag: str, now: float) -> None:
        live_req_ids = {
            req.request_id
            for req in self.requests.values()
            if req.status != RequestStatus.COMPLETED
        }

        for taxi_id, plan in self.taxi_plans.items():
            bad = [s.request_id for s in plan.stops if s.request_id not in live_req_ids]
            if bad:
                print(f"\n🚨 [BUG DETECTED] at {tag} (t={now:.1f})")
                print(f"Taxi: {taxi_id}")
                print(f"Bad stops: {bad}")
                print(f"Current stops: {[s.request_id for s in plan.stops]}")
                print(f"Live requests: {sorted(live_req_ids)}")
                for rid in bad:
                    r = self.requests.get(rid)
                    print(f"  → req {rid}: {r.status if r else 'NOT IN self.requests'}")
                print("-----")