from __future__ import annotations

import argparse
import json
from pathlib import Path

from DRTDataclass import RequestStatus
from dispatcher import setup_logger
from dqn_env import DQNStepEnvironment
from dqn_policy import DQNPolicy


def summarize_run(env: DQNStepEnvironment) -> dict:
    requests = list(env.requests.values())
    total = len(requests)
    completed = [r for r in requests if r.status == RequestStatus.COMPLETED]
    picked_up = [r for r in requests if r.pickup_time is not None]
    dropped = [r for r in completed if r.dropoff_time is not None and r.pickup_time is not None]

    avg_wait = sum((r.pickup_time - r.request_time) for r in picked_up) / len(picked_up) if picked_up else 0.0
    max_wait = max(((r.pickup_time - r.request_time) for r in picked_up), default=0.0)
    avg_excess = sum((r.excess_ride_time or 0.0) for r in dropped) / len(dropped) if dropped else 0.0

    return {
        "total_requests": total,
        "completed_requests": len(completed),
        "completion_rate": (len(completed) / total) if total else 0.0,
        "picked_up_requests": len(picked_up),
        "avg_wait_until_pickup": avg_wait,
        "max_wait_until_pickup": max_wait,
        "avg_excess_ride_time": avg_excess,
        "decisions_seen": getattr(env, "_decision_counter", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained DQN policy in the single-decision DQN environment.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--summary-out", default="dqn_run_summary.json")
    args = parser.parse_args()

    setup_logger("dqn_policy_run.log")
    policy = DQNPolicy(model_dir=args.model_dir, device=args.device, print_top_k=False, epsilon=0.0)
    env = DQNStepEnvironment(cfg_path=args.cfg, step_length=args.step_length, use_gui=args.gui, policy=None, dataset_logger=None, verbose=False)

    try:
        decision = env.reset_episode()
        while decision is not None:
            output = policy.select_action(decision, env.taxi_plans, decision.sim_time)
            chosen_idx = next(i for i, ev in enumerate(output.evaluations) if ev.chosen)
            step_result = env.step_decision(chosen_idx)
            decision = None if step_result.done else step_result.next_decision

        summary = summarize_run(env)
        print(json.dumps(summary, indent=2))
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.close_episode()


if __name__ == "__main__":
    main()
