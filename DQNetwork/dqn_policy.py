from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from drt_policy_types import CandidateEvaluation, DecisionPoint, PolicyOutput
from feature_extractor import flatten_decision_features
from heuristic_policy import BasePolicy
from q_network import ParametricQNetwork, TaxiFairQNetwork


class DQNPolicy(BasePolicy):
    name = "dqn"

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        print_top_k: bool = True,
        epsilon: float = 0.0,
        forbid_defer_when_action_exists: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.print_top_k = print_top_k
        self.epsilon = float(epsilon)
        self.forbid_defer_when_action_exists = forbid_defer_when_action_exists
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        metadata_path = self.model_dir / "dqn_metadata.json"
        model_path = self.model_dir / "dqn_model.pt"
        scaler_path = self.model_dir / "feature_scaler.joblib"

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)
        self.feature_columns = list(self.metadata["feature_columns"])
        hidden_dims = list(self.metadata.get("hidden_dims", [256, 128]))
        dropout = float(self.metadata.get("dropout", 0.1))
        input_dim = int(self.metadata.get("input_dim", len(self.feature_columns)))

        use_taxi_fair = bool(self.metadata.get("use_taxi_fair", False))
        self.use_taxi_fair = use_taxi_fair

        self.scaler = joblib.load(scaler_path)
        if use_taxi_fair:
            self.model = TaxiFairQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        else:
            self.model = ParametricQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _build_feature_matrix(self, decision_point: DecisionPoint, taxi_plans) -> np.ndarray:
        rows: list[list[float]] = []
        seen_taxis: dict[str, int] = {}
        group_ids: list[float] = []

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
                raise KeyError(f"Missing feature columns at inference: {missing[:10]}")
            rows.append([float(feature_dict[col]) for col in self.feature_columns])

            if cand.is_defer:
                group_ids.append(-1.0)
            else:
                tid = cand.taxi_id
                if tid not in seen_taxis:
                    seen_taxis[tid] = len(seen_taxis)
                group_ids.append(float(seen_taxis[tid]))

        x = np.asarray(rows, dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32)
        valid_mask = np.ones((x.shape[0], 1), dtype=np.float32)

        if self.use_taxi_fair:
            group_col = np.array(group_ids, dtype=np.float32).reshape(-1, 1)
            return np.concatenate([x, group_col, valid_mask], axis=1)
        return np.concatenate([x, valid_mask], axis=1)

    @torch.no_grad()
    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        x = self._build_feature_matrix(decision_point, taxi_plans)
        inp = torch.from_numpy(x[None, :, :]).to(self.device)
        q_values, _ = self.model(inp)
        scores = q_values[0].detach().cpu().numpy().astype(float)

        if self.forbid_defer_when_action_exists and any(not c.is_defer for c in decision_point.candidate_actions):
            for i, cand in enumerate(decision_point.candidate_actions):
                if cand.is_defer:
                    scores[i] = -1e9

        order = np.argsort(-scores)
        if self.epsilon > 0 and random.random() < self.epsilon:
            valid = [i for i, cand in enumerate(decision_point.candidate_actions) if scores[i] > -1e8]
            chosen_idx = random.choice(valid)
        else:
            chosen_idx = int(order[0])

        evaluations: list[CandidateEvaluation] = []
        for idx, (cand, score) in enumerate(zip(decision_point.candidate_actions, scores.tolist())):
            evaluations.append(CandidateEvaluation(candidate=cand, score=float(score), chosen=(idx == chosen_idx), policy_name=self.name))
        for rank, idx in enumerate(order, start=1):
            evaluations[idx].rank = rank

        return PolicyOutput(chosen_action=decision_point.candidate_actions[chosen_idx], evaluations=evaluations, policy_name=self.name)
