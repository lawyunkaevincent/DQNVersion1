from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dispatcher import setup_logger
from dataset_logger import ImitationDatasetLogger
from dispatcher_env import RefactoredDRTEnvironment
from heuristic_policy import HeuristicPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect imitation-learning dataset from heuristic dispatcher")
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length")
    parser.add_argument(
        "--dataset-out",
        default="imitation_dataset.csv",
        help="CSV path for one-row-per-candidate imitation dataset",
    )
    args = parser.parse_args()
    

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(f"imitation_collect_{ts}.log")

    dataset_logger = ImitationDatasetLogger(args.dataset_out)
    env = RefactoredDRTEnvironment(
        cfg_path=args.cfg,
        step_length=args.step_length,
        use_gui=args.gui,
        policy=HeuristicPolicy(print_top_k=True),
        dataset_logger=dataset_logger,
    )

    try:
        env.start()
        env.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
