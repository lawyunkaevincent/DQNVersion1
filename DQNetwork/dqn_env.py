from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import traci

from DRTDataclass import CandidateInsertion, RequestStatus, TickContext, TickOutcome
from dispatcher import (
    _detect_events,
    _eligible_taxis_for_tick,
    _print_tick_summary,
    _refresh_taxi_plans,
    _sync_onboard,
)
from dispatcher_env import RefactoredDRTEnvironment
from drt_policy_types import DecisionPoint


@dataclass
class StepResult:
    next_decision: DecisionPoint | None
    reward: float
    done: bool
    info: dict


class DQNStepEnvironment(RefactoredDRTEnvironment):
    """Single-decision-per-meaningful-tick environment for DQN.

    This deliberately simplifies the original multi-request-per-tick dispatcher.
    Each RL action corresponds to exactly one request decision, making replay
    transitions well-defined.
    """

    def __init__(self, *args, reward_weights: dict | None = None, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.reward_weights = reward_weights or {}
        self.current_decision: DecisionPoint | None = None

    def reset_episode(self) -> DecisionPoint | None:
        self.start()
        decision, done = self._advance_until_next_decision()
        self.current_decision = decision
        if done:
            return None
        return decision

    def step_decision(self, action_index: int) -> StepResult:
        if self.current_decision is None:
            raise RuntimeError("step_decision called with no active decision.")
        if action_index < 0 or action_index >= len(self.current_decision.candidate_actions):
            raise IndexError(f"Invalid action index {action_index} for {len(self.current_decision.candidate_actions)} candidates")

        chosen = self.current_decision.candidate_actions[action_index]
        request = self.current_decision.request
        now = self.current_decision.sim_time
        self.apply_action(request, chosen, now)
        self._flush_idle_dispatches()
        self.accumulator.reset()

        next_decision, done = self._advance_until_next_decision()
        reward = self.accumulator.compute_reward(**self.reward_weights)
        info = {
            "decision_id": self.current_decision.decision_id,
            "request_id": self.current_decision.request.request_id,
            "chosen_is_defer": bool(chosen.is_defer),
            "reward": reward,
        }
        self.current_decision = next_decision
        return StepResult(next_decision=next_decision, reward=reward, done=done, info=info)

    def close_episode(self) -> None:
        self.close()

    def _advance_until_next_decision(self) -> tuple[DecisionPoint | None, bool]:
        while True:
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                return None, True

            try:
                now = traci.simulation.getTime()
            except traci.exceptions.FatalTraCIError:
                return None, True

            self._step_count += 1

            dt = self.step_length
            pending_count = sum(
                1 for r in self.requests.values()
                if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)
            )
            self.accumulator.wait_cost += pending_count * dt
            self.accumulator.elapsed_time += dt

            _refresh_taxi_plans(self.taxi_plans)

            try:
                active_vids = set(traci.vehicle.getIDList())
            except traci.TraCIException:
                active_vids = set()

            # Remove taxis that have already left the live simulation.
            stale_taxis = [tid for tid in list(self.taxi_plans.keys()) if tid not in active_vids]
            for tid in stale_taxis:
                self.taxi_plans.pop(tid, None)
                self._pending_dispatches.discard(tid)
                self._dispatch_snapshots.pop(tid, None)

            for taxi_id, plan in list(self.taxi_plans.items()):
                try:
                    dist = traci.vehicle.getDistance(taxi_id)
                    delta = dist - plan.cumulative_distance
                    plan.cumulative_distance = dist
                    if plan.onboard_count == 0 and delta > 0:
                        self.accumulator.empty_dist_cost += delta
                except traci.TraCIException:
                    # Taxi disappeared between getIDList() and getDistance().
                    self.taxi_plans.pop(taxi_id, None)
                    self._pending_dispatches.discard(taxi_id)
                    self._dispatch_snapshots.pop(taxi_id, None)

            if self._step_count >= self.TICK_STEPS:
                self._step_count = 0
                self._tick_num += 1
                decision = self._process_tick_for_step(now)
                if decision is not None:
                    return decision, False

            try:
                if self._termination_ready():
                    return None, True
            except traci.exceptions.FatalTraCIError:
                return None, True

    def _process_tick_for_step(self, now: float) -> DecisionPoint | None:
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

        try:
            active_pids = set(traci.person.getIDList())
        except Exception:
            active_pids = set(self.requests.keys())

        pending_pool = [
            pid for pid, r in self.requests.items()
            if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)
            and pid in active_pids
        ]
        self._eligible_taxis_this_tick = _eligible_taxis_for_tick(
            self.taxi_plans, self.requests, new_pickups, new_dropoffs,
        )

        chosen_request = None
        if pending_pool and self._eligible_taxis_this_tick:
            sorted_pool = sorted(
                pending_pool,
                key=lambda pid: self.requests[pid].waiting_time(now),
                reverse=True,
            )
            for pid in sorted_pool:
                req = self.requests[pid]
                cands = self.build_candidates_for_request(req, now)
                if any(not c.is_defer for c in cands):
                    chosen_request = req
                    break

        has_candidates = chosen_request is not None
        outcome = TickOutcome.MEANINGFUL if has_candidates else TickOutcome.IDLE
        tick = TickContext(
            outcome=outcome,
            pending_pool=pending_pool,
            has_candidates=has_candidates,
            sim_time=now,
        )

        if self.verbose and (had_event or has_candidates):
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

        self._prev_req_ids = set(self.requests.keys())
        self._prev_onboard_ids = {pid for pid, r in self.requests.items() if r.status == RequestStatus.ONBOARD}
        self._prev_completed_ids = {pid for pid, r in self.requests.items() if r.status == RequestStatus.COMPLETED}

        # CRITICAL: flush any pending dispatches on IDLE ticks too.
        #
        # Without this, _pending_dispatches accumulates entries across multiple
        # IDLE ticks while SUMO advances (completing pickups / dropoffs). By the
        # time step_decision() calls _flush_idle_dispatches(), the reservation
        # IDs in those pending plans are already closed in SUMO, causing the
        # "Reservation id 'X' is not known" error.
        #
        # On MEANINGFUL ticks we do NOT flush here — step_decision() owns the
        # flush so it can apply the agent's chosen action first, then flush once.
        if outcome == TickOutcome.IDLE and self._pending_dispatches:
            self._flush_idle_dispatches()

        if not has_candidates:
            return None
        return self.build_decision_point(chosen_request, now, tick_context=tick)