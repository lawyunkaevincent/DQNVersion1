from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import traci

from DRTDataclass import (
    CandidateInsertion,
    GlobalStateSummary,
    Request,
    RequestStatus,
    Stop,
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
    _refresh_taxi_plans,
    _sync_onboard,
    generate_candidates,
)
from drt_policy_types import DecisionPoint, PolicyOutput
from heuristic_policy import BasePolicy, HeuristicPolicy
from dataset_logger import ImitationDatasetLogger


class RefactoredDRTEnvironment(LegacyHeuristicDispatcher):
    """
    Agent-driven wrapper around the original dispatcher.

    Main refactor points:
      - build_global_state_summary()
      - build_candidates_for_request()
      - build_decision_point()
      - apply_action()
      - process one policy decision instead of hardcoding selection inside _dispatch_best()
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
    # New refactored decision helpers
    # ------------------------------------------------------------------

    def build_global_state_summary(self, now: float) -> GlobalStateSummary:
        pending = [r for r in self.requests.values() if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]
        onboard_count = sum(plan.onboard_count for plan in self.taxi_plans.values())
        idle_taxi_count = sum(1 for plan in self.taxi_plans.values() if plan.status == TaxiStatus.IDLE and plan.onboard_count == 0)
        active_taxi_count = sum(1 for plan in self.taxi_plans.values() if not (plan.status == TaxiStatus.IDLE and len(plan.stops) == 0 and plan.onboard_count == 0))
        avg_wait = (sum(r.waiting_time(now) for r in pending) / len(pending)) if pending else 0.0
        max_wait = max((r.waiting_time(now) for r in pending), default=0.0)

        taxi_count = max(1, len(self.taxi_plans))
        avg_occupancy = sum((plan.onboard_count / max(1, plan.capacity)) for plan in self.taxi_plans.values()) / taxi_count
        fleet_utilization = sum(1 for plan in self.taxi_plans.values() if plan.status != TaxiStatus.IDLE or plan.onboard_count > 0) / taxi_count

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

    def build_decision_point(self, request: Request, now: float, tick_context: Optional[TickContext] = None) -> DecisionPoint | None:
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

        plan.stops = action.resulting_stops
        plan.assigned_request_ids.add(request.person_id)
        request.assigned_taxi_id = action.taxi_id
        request.status = RequestStatus.ASSIGNED
        self._pending_dispatches.add(action.taxi_id)

        _log(
            f"  → ASSIGNED req={request.request_id} → taxi={action.taxi_id} "
            f" pu_eta={action.pickup_eta_new:.1f}s  (flush pending)"
        )
        return action

    def dispatch_request_via_policy(self, request: Request, now: float, tick_context: Optional[TickContext] = None) -> PolicyOutput | None:
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
            self.taxi_plans, self.requests, new_pickups, new_dropoffs,
        )

        # _log(f"PENDING POOL: {pending_pool}")

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
