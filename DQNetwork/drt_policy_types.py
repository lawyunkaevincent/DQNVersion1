from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from DRTDataclass import CandidateInsertion, GlobalStateSummary, Request, TickContext


@dataclass
class DecisionPoint:
    """One agent decision for one request at one tick."""
    request: Request
    state_summary: GlobalStateSummary
    candidate_actions: list[CandidateInsertion]
    sim_time: float
    tick_context: Optional[TickContext] = None
    decision_id: str = ""


@dataclass
class CandidateEvaluation:
    """Scored candidate metadata used by policies and dataset logging."""
    candidate: CandidateInsertion
    score: float
    chosen: bool = False
    rank: int = -1
    policy_name: str = ""


@dataclass
class PolicyOutput:
    """Result returned by a policy after scoring a decision point."""
    chosen_action: CandidateInsertion
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    policy_name: str = ""
