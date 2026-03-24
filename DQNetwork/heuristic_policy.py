from __future__ import annotations

from abc import ABC, abstractmethod

from dispatcher import score_candidate
from drt_policy_types import CandidateEvaluation, DecisionPoint, PolicyOutput


class BasePolicy(ABC):
    name = "base_policy"

    @abstractmethod
    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        raise NotImplementedError


class HeuristicPolicy(BasePolicy):
    name = "heuristic"

    def __init__(self, print_top_k: bool = True):
        self.print_top_k = print_top_k

    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        evaluations: list[CandidateEvaluation] = []
        for cand in decision_point.candidate_actions:
            score = score_candidate(cand, decision_point.request, taxi_plans, now)
            evaluations.append(
                CandidateEvaluation(candidate=cand, score=score, policy_name=self.name)
            )

        if not evaluations:
            raise ValueError("HeuristicPolicy received a decision point with no candidates.")

        order = sorted(range(len(evaluations)), key=lambda i: evaluations[i].score, reverse=True)
        for rank, idx in enumerate(order, start=1):
            evaluations[idx].rank = rank

        best_idx = order[0]
        evaluations[best_idx].chosen = True

        return PolicyOutput(
            chosen_action=evaluations[best_idx].candidate,
            evaluations=evaluations,
            policy_name=self.name,
        )
