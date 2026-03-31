"""
Reward shaping for DQN training.

PRIMARY objective: minimise avg_wait_until_pickup and avg_excess_ride_time.
All other metrics are secondary.

Key bug fixed in compute_shaped_reward_v2 (v2 revision):
    The quality_bonus previously looped over ALL requests in requests_dict
    and credited every historically-completed passenger at every decision step.
    With 200 requests, by step 150 the bonus was ~150 × 1.5 × 0.7 ≈ 157 per
    step — completely overwhelming the wait/ride penalties and allowing the
    agent to achieve high reward while wait times actually increased.
    Fix: use accumulator.completed_dropoffs (new completions this interval
    only) for the bonus, matching the accumulator's reset semantics.

Signal priorities in compute_shaped_reward_v2:
    1. wait_penalty  — coefficient 1.5 (tripled)  — PRIMARY
    2. ride_penalty  — coefficient 0.6 (doubled)   — PRIMARY
    3. action_penalty (predicted wait/ride)         — SECONDARY
    4. quality_bonus (per new completion)           — SECONDARY
    5. defer_penalty, empty_penalty                 — minor signals
"""
from __future__ import annotations

import math


def compute_shaped_reward_v2(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
    chosen_candidate=None,
    request=None,
    requests_dict=None,  # kept for API compatibility but no longer used
) -> float:
    """
    Reward function focused on minimising avg_wait_until_pickup and
    avg_excess_ride_time as the PRIMARY objectives.

    Key fixes vs the previous version:
      - quality_bonus now uses accumulator.completed_dropoffs (new dropoffs
        this interval ONLY) instead of re-counting all historical completions
        from requests_dict on every step — that bug inflated rewards and
        caused high-reward episodes to still have high wait times.
      - wait_penalty coefficient tripled (1.5) to dominate the reward signal.
      - ride_penalty coefficient doubled (0.6) as second-priority signal.
      - action_penalty for predicted excess wait strengthened (0.005).
    """
    t = max(elapsed_time, 1.0)

    # ────────────────────────────────────────────────────────
    # 1. WAIT PENALTY — PRIMARY dominant signal
    # ────────────────────────────────────────────────────────
    # passenger-seconds of waiting, normalized by time interval.
    # Coefficient tripled (1.5 vs old 0.5) so this term dominates.
    raw_wait_rate = accumulator.wait_cost / t
    wait_penalty = 1.5 * math.sqrt(max(0.0, raw_wait_rate))

    # ────────────────────────────────────────────────────────
    # 2. RIDE / DETOUR PENALTY — PRIMARY signal
    # ────────────────────────────────────────────────────────
    # excess ride time for passengers dropped off this interval.
    # Coefficient doubled (0.6 vs old 0.3).
    raw_ride_rate = accumulator.ride_cost / t
    ride_penalty = 0.6 * math.sqrt(max(0.0, raw_ride_rate))

    # ────────────────────────────────────────────────────────
    # 3. CHOSEN-ACTION QUALITY PENALTY
    #    Direct penalty based on what the agent just decided
    # ────────────────────────────────────────────────────────
    action_penalty = 0.0
    if chosen_candidate is not None and not getattr(chosen_candidate, 'is_defer', True):
        # Penalize predicted wait time for the new passenger (PRIMARY)
        if request is not None:
            predicted_wait = max(0.0, chosen_candidate.pickup_eta_new - request.request_time)
            # Penalty once wait exceeds 120s baseline — strengthened (0.005 vs old 0.002)
            excess_wait = max(0.0, predicted_wait - 120.0)
            action_penalty += 0.005 * excess_wait

        # Penalize predicted detour for the new passenger (PRIMARY)
        if request is not None and request.direct_travel_time > 0:
            predicted_ride = chosen_candidate.dropoff_eta_new - chosen_candidate.pickup_eta_new
            excess_ride = max(0.0, predicted_ride - request.direct_travel_time)
            action_penalty += 0.002 * excess_ride

        # Penalize delay imposed on existing passengers (SECONDARY — reduced)
        action_penalty += 0.001 * chosen_candidate.max_existing_delay

        # Constraint violation penalties (squared for emphasis)
        action_penalty += 0.001 * (chosen_candidate.new_wait_violation ** 2) / 100.0
        action_penalty += 0.001 * (chosen_candidate.new_ride_violation ** 2) / 100.0
        action_penalty += 0.002 * (chosen_candidate.existing_wait_violation_sum ** 2) / 100.0
        action_penalty += 0.002 * (chosen_candidate.existing_ride_violation_sum ** 2) / 100.0

    # ────────────────────────────────────────────────────────
    # 4. COMPLETION BONUS
    #    BUG FIX: use accumulator.completed_dropoffs (NEW dropoffs this
    #    interval only) instead of looping all requests_dict — the old loop
    #    re-counted every historically-completed passenger at each step,
    #    inflating the bonus and masking the wait/ride penalties.
    # ────────────────────────────────────────────────────────
    quality_bonus = 1.0 * accumulator.completed_dropoffs

    # ────────────────────────────────────────────────────────
    # 5. DEFER PENALTY
    # ────────────────────────────────────────────────────────
    defer_penalty = 0.5 if chosen_is_defer else 0.0

    # ────────────────────────────────────────────────────────
    # 6. EMPTY DRIVING (small informational signal)
    # ────────────────────────────────────────────────────────
    raw_empty = accumulator.empty_dist_cost / t
    empty_penalty = 0.0005 * min(raw_empty, 100.0)

    reward = quality_bonus - wait_penalty - ride_penalty - action_penalty - defer_penalty - empty_penalty
    return reward


def compute_shaped_reward(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
) -> float:
    """
    ORIGINAL reward function — kept for backward compatibility.
    Use compute_shaped_reward_v2 for new training runs.
    """
    t = max(elapsed_time, 1.0)

    raw_wait = accumulator.wait_cost / t
    wait_penalty = 0.3 * min(raw_wait, 10.0)

    raw_ride = accumulator.ride_cost / t
    ride_penalty = 0.2 * min(raw_ride, 10.0)

    raw_empty = accumulator.empty_dist_cost / t
    empty_penalty = 0.0005 * min(raw_empty, 100.0)

    defer_penalty = 0.3 if chosen_is_defer else 0.0

    completion_bonus = 2.0 * accumulator.completed_dropoffs

    return completion_bonus - wait_penalty - ride_penalty - empty_penalty - defer_penalty