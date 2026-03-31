"""
Improved reward shaping for DQN training.

Key problems with the old compute_shaped_reward:
    1. completion_bonus = 2.0 * completed_dropoffs dominates everything.
       Since all 200 requests complete every episode, the bonus is always
       ~400 total, drowning out wait/ride penalties.
    2. wait_penalty is capped at 3.0 — a passenger waiting 700s produces
       the same penalty as one waiting 100s.
    3. ride_cost is also capped too aggressively.
    4. The penalty terms are normalized by elapsed_time, which varies wildly
       between decisions (10s to 200s), making the signal noisy.

New design principles:
    1. Remove the per-decision completion bonus since it's constant.
       Replace with a per-dropoff quality bonus that rewards LOW wait
       and LOW detour for the specific passengers who completed.
    2. Make wait penalty scale with actual passenger-seconds, no cap.
       Use sqrt scaling to compress but not clip extreme values.
    3. Add an explicit per-decision detour penalty based on the chosen
       candidate's predicted excess ride time.
    4. Add a penalty proportional to how many constraint violations
       the chosen action creates.
"""
from __future__ import annotations

import math


def compute_shaped_reward_v2(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
    chosen_candidate=None,
    request=None,
    requests_dict=None,
) -> float:
    """
    Improved reward function focused on minimizing wait time and detour.

    Reward range per decision: roughly [-5.0, +3.0]
    """
    t = max(elapsed_time, 1.0)

    # ────────────────────────────────────────────────────────
    # 1. WAIT PENALTY — the dominant signal for your goal
    # ────────────────────────────────────────────────────────
    # passenger-seconds of waiting, normalized by time interval
    # Use sqrt to compress but NOT clip — 700s wait should hurt
    # much more than 100s wait
    raw_wait_rate = accumulator.wait_cost / t
    wait_penalty = 0.5 * math.sqrt(max(0.0, raw_wait_rate))

    # ────────────────────────────────────────────────────────
    # 2. RIDE / DETOUR PENALTY
    # ────────────────────────────────────────────────────────
    # excess ride time for passengers dropped off this interval
    raw_ride_rate = accumulator.ride_cost / t
    ride_penalty = 0.3 * math.sqrt(max(0.0, raw_ride_rate))

    # ────────────────────────────────────────────────────────
    # 3. CHOSEN-ACTION QUALITY PENALTY
    #    Direct penalty based on what the agent just decided
    # ────────────────────────────────────────────────────────
    action_penalty = 0.0
    if chosen_candidate is not None and not getattr(chosen_candidate, 'is_defer', True):
        # Penalize predicted wait time for the new passenger
        if request is not None:
            predicted_wait = max(0.0, chosen_candidate.pickup_eta_new - request.request_time)
            # Quadratic penalty once wait exceeds 120s baseline
            excess_wait = max(0.0, predicted_wait - 120.0)
            action_penalty += 0.002 * excess_wait

        # Penalize predicted detour for the new passenger
        if request is not None and request.direct_travel_time > 0:
            predicted_ride = chosen_candidate.dropoff_eta_new - chosen_candidate.pickup_eta_new
            excess_ride = max(0.0, predicted_ride - request.direct_travel_time)
            action_penalty += 0.001 * excess_ride

        # Penalize delay imposed on existing passengers
        action_penalty += 0.003 * chosen_candidate.max_existing_delay

        # Constraint violation penalties (squared for emphasis)
        action_penalty += 0.001 * (chosen_candidate.new_wait_violation ** 2) / 100.0
        action_penalty += 0.001 * (chosen_candidate.new_ride_violation ** 2) / 100.0
        action_penalty += 0.002 * (chosen_candidate.existing_wait_violation_sum ** 2) / 100.0
        action_penalty += 0.002 * (chosen_candidate.existing_ride_violation_sum ** 2) / 100.0

    # ────────────────────────────────────────────────────────
    # 4. COMPLETION QUALITY BONUS (replaces flat +2.0 per dropoff)
    #    Only rewards dropoffs with good service quality
    # ────────────────────────────────────────────────────────
    quality_bonus = 0.0
    if accumulator.completed_dropoffs > 0 and requests_dict is not None:
        # Give bonus scaled by how well each completed passenger was served
        for pid, req in requests_dict.items():
            if req.status.name != "COMPLETED":
                continue
            if req.pickup_time is None or req.dropoff_time is None:
                continue

            actual_wait = req.pickup_time - req.request_time
            max_wait = getattr(req, 'max_wait', 300.0)

            # Bonus: +1.0 for perfect service, scaled down for worse service
            wait_quality = max(0.0, 1.0 - actual_wait / max(max_wait, 1.0))

            excess = req.excess_ride_time or 0.0
            dtt = max(req.direct_travel_time, 1.0)
            ride_quality = max(0.0, 1.0 - excess / dtt)

            # Combined quality score [0, 1]
            q = 0.6 * wait_quality + 0.4 * ride_quality
            quality_bonus += 1.5 * q

    # Use a simpler completion bonus when we don't have request details
    if quality_bonus == 0.0:
        quality_bonus = 0.5 * accumulator.completed_dropoffs

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