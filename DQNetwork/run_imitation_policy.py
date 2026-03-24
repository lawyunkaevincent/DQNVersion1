from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dispatcher import setup_logger
from dispatcher_env import RefactoredDRTEnvironment
from DRTDataclass import RequestStatus
from imitation_policy import ImitationPolicy


def summarize_run(env: RefactoredDRTEnvironment) -> dict:
    requests = list(env.requests.values())
    total = len(requests)
    completed = [r for r in requests if r.status == RequestStatus.COMPLETED]
    onboard = [r for r in requests if r.status == RequestStatus.ONBOARD]
    assigned = [r for r in requests if r.status == RequestStatus.ASSIGNED]
    pending = [r for r in requests if r.status in (RequestStatus.PENDING, RequestStatus.DEFERRED)]

    picked_up = [r for r in requests if r.pickup_time is not None]
    dropped = [r for r in completed if r.dropoff_time is not None and r.pickup_time is not None]

    avg_wait_completed = (
        sum((r.pickup_time - r.request_time) for r in picked_up) / len(picked_up)
        if picked_up else 0.0
    )
    max_wait_completed = max(((r.pickup_time - r.request_time) for r in picked_up), default=0.0)
    avg_excess_ride = (
        sum((r.excess_ride_time or 0.0) for r in dropped) / len(dropped)
        if dropped else 0.0
    )
    max_excess_ride = max(((r.excess_ride_time or 0.0) for r in dropped), default=0.0)
    current_sim_time = max((r.dropoff_time or r.pickup_time or r.request_time for r in requests), default=0.0)

    return {
        "total_requests": total,
        "completed_requests": len(completed),
        "completion_rate": (len(completed) / total) if total else 0.0,
        "picked_up_requests": len(picked_up),
        "currently_onboard": len(onboard),
        "currently_assigned": len(assigned),
        "still_pending_or_deferred": len(pending),
        "avg_wait_until_pickup": avg_wait_completed,
        "max_wait_until_pickup": max_wait_completed,
        "avg_excess_ride_time": avg_excess_ride,
        "max_excess_ride_time": max_excess_ride,
        "decisions_seen": getattr(env, "_decision_counter", 0),
        "final_accum_completed_dropoffs": getattr(env.accumulator, "completed_dropoffs", 0),
        "final_accum_wait_cost": getattr(env.accumulator, "wait_cost", 0.0),
        "final_accum_ride_cost": getattr(env.accumulator, "ride_cost", 0.0),
        "last_known_request_time": current_sim_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the refactored dispatcher using a trained imitation policy.")
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--model-dir", required=True, help="Directory containing imitation_model.pt and metadata")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length")
    parser.add_argument("--device", default=None, help="cpu or cuda; default auto")
    parser.add_argument(
        "--allow-defer-even-when-action-exists",
        action="store_true",
        help="By default DEFER is masked whenever a real candidate exists.",
    )
    parser.add_argument(
        "--heuristic-fallback-gap",
        type=float,
        default=0.25,
        help="If model top-1 minus top-2 score gap is smaller than this, use the heuristic instead. Use a negative value to disable fallback.",
    )
    parser.add_argument("--summary-out", default="imitation_run_summary.json", help="JSON file to save run summary")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f"imitation_policy_run_{ts}.log")

    fallback_gap = None if args.heuristic_fallback_gap < 0 else args.heuristic_fallback_gap
    policy = ImitationPolicy(
        model_dir=args.model_dir,
        device=args.device,
        print_top_k=True,
        forbid_defer_when_action_exists=not args.allow_defer_even_when_action_exists,
        heuristic_fallback_gap=fallback_gap,
    )
    env = RefactoredDRTEnvironment(
        cfg_path=args.cfg,
        step_length=args.step_length,
        use_gui=args.gui,
        policy=policy,
        dataset_logger=None,
    )

    try:
        env.start()
        env.run()
    finally:
        try:
            summary = summarize_run(env)
            print("\nRun summary")
            print(json.dumps(summary, indent=2))
            out_path = Path(args.summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        finally:
            env.close()


if __name__ == "__main__":
    main()
