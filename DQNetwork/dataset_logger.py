from __future__ import annotations

import csv
from pathlib import Path

from drt_policy_types import DecisionPoint, PolicyOutput
from feature_extractor import flatten_decision_features


class ImitationDatasetLogger:
    """Logs one row per candidate for imitation learning.

    Important design choice:
    - keep only non-feature metadata columns here
    - all model features should come from flatten_decision_features(...)

    This avoids duplicated columns like candidate_is_defer vs cand_is_defer,
    which can silently create train/inference mismatches.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self._header_written = self.output_path.exists() and self.output_path.stat().st_size > 0

    def log_decision(self, decision_point: DecisionPoint, policy_output: PolicyOutput, taxi_plans) -> None:
        rows: list[dict] = []
        for eval_ in policy_output.evaluations:
            row = {
                "decision_id": decision_point.decision_id,
                "policy_name": policy_output.policy_name,
                "request_id": decision_point.request.request_id,
                "person_id": decision_point.request.person_id,
                "candidate_taxi_id": eval_.candidate.taxi_id,
                "chosen": int(eval_.chosen),
                "rank": int(eval_.rank),
                "heuristic_score": float(eval_.score),
            }
            row.update(
                flatten_decision_features(
                    decision_point.state_summary,
                    decision_point.request,
                    eval_.candidate,
                    taxi_plans,
                    decision_point.sim_time,
                )
            )
            rows.append(row)

        self._append_rows(rows)

    def _append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)
