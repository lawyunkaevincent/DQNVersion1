from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from drt_policy_types import CandidateEvaluation, DecisionPoint, PolicyOutput
from feature_extractor import flatten_decision_features
from heuristic_policy import BasePolicy, HeuristicPolicy
from train_imitation_model import ImitationRanker


class ImitationPolicy(BasePolicy):
    """Inference policy for the trained imitation ranker.

    Improvements over the first version:
    - strict feature-column matching
    - optional DEFER masking whenever at least one real action exists
    - optional heuristic fallback when the model is uncertain
    """

    name = "imitation_v2"

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        print_top_k: bool = True,
        forbid_defer_when_action_exists: bool = True,
        heuristic_fallback_gap: float | None = 0.25,
    ):
        self.model_dir = Path(model_dir)
        self.print_top_k = print_top_k
        self.forbid_defer_when_action_exists = forbid_defer_when_action_exists
        self.heuristic_fallback_gap = heuristic_fallback_gap
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.heuristic_policy = HeuristicPolicy(print_top_k=False)

        metadata_path = self.model_dir / "model_metadata.json"
        model_path = self.model_dir / "imitation_model.pt"
        scaler_path = self.model_dir / "feature_scaler.joblib"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.feature_columns: list[str] = list(self.metadata["feature_columns"])
        hidden_dims = list(self.metadata.get("hidden_dims", [256, 128]))
        dropout = float(self.metadata.get("dropout", 0.1))
        input_dim = int(self.metadata.get("input_dim", len(self.feature_columns)))

        self.scaler = joblib.load(scaler_path)
        self.model = ImitationRanker(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _build_feature_matrix(self, decision_point: DecisionPoint, taxi_plans) -> np.ndarray:
        rows: list[list[float]] = []
        for cand in decision_point.candidate_actions:
            feature_dict = flatten_decision_features(
                decision_point.state_summary,
                decision_point.request,
                cand,
                taxi_plans,
                decision_point.sim_time,
            )

            missing = [col for col in self.feature_columns if col not in feature_dict]
            if missing:
                preview = ", ".join(missing[:10])
                raise KeyError(
                    "Missing feature columns at inference. "
                    f"First missing columns: {preview}"
                )

            row = [float(feature_dict[col]) for col in self.feature_columns]
            rows.append(row)

        if not rows:
            raise ValueError("Decision point has no candidate actions.")

        x = np.asarray(rows, dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32)
        return x

    def _apply_defer_mask(self, scores: np.ndarray, decision_point: DecisionPoint) -> np.ndarray:
        scores = scores.copy()
        if self.forbid_defer_when_action_exists:
            has_non_defer = any(not cand.is_defer for cand in decision_point.candidate_actions)
            if has_non_defer:
                for i, cand in enumerate(decision_point.candidate_actions):
                    if cand.is_defer:
                        scores[i] = -1e9
        return scores

    def _build_policy_output(self, decision_point: DecisionPoint, scores: np.ndarray, policy_name: str) -> PolicyOutput:
        evaluations: list[CandidateEvaluation] = []
        for cand, score in zip(decision_point.candidate_actions, scores.tolist()):
            evaluations.append(
                CandidateEvaluation(candidate=cand, score=float(score), policy_name=policy_name)
            )

        order = sorted(range(len(evaluations)), key=lambda i: evaluations[i].score, reverse=True)
        for rank, idx in enumerate(order, start=1):
            evaluations[idx].rank = rank

        best_idx = order[0]
        evaluations[best_idx].chosen = True
        return PolicyOutput(
            chosen_action=evaluations[best_idx].candidate,
            evaluations=evaluations,
            policy_name=policy_name,
        )

    @torch.no_grad()
    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        x = self._build_feature_matrix(decision_point, taxi_plans)
        valid_mask = np.ones((x.shape[0], 1), dtype=np.float32)
        x_with_mask = np.concatenate([x, valid_mask], axis=1)

        inp = torch.from_numpy(x_with_mask[None, :, :]).to(self.device)
        logits, _mask = self.model(inp)
        scores = logits[0].detach().cpu().numpy().astype(float)
        scores = self._apply_defer_mask(scores, decision_point)

        order = np.argsort(-scores)
        gap = None
        if len(order) >= 2:
            gap = float(scores[order[0]] - scores[order[1]])

        if self.heuristic_fallback_gap is not None and gap is not None and gap < self.heuristic_fallback_gap:
            heuristic_output = self.heuristic_policy.select_action(decision_point, taxi_plans, now)
            heuristic_output.policy_name = f"{self.name}_fallback_heuristic"
            for ev in heuristic_output.evaluations:
                ev.policy_name = heuristic_output.policy_name
            return heuristic_output

        return self._build_policy_output(decision_point, scores, self.name)
