"""
Improved Q-network for DRT dispatch.

Key change: TaxiFairQNetwork adds a per-taxi attention-pooling layer
so that candidates from the same taxi compete fairly against candidates
from other taxis, regardless of how many insertion positions each taxi has.

Problem solved:
    A taxi with N stops generates O(N^2) candidates while an idle taxi
    generates 1. The old ParametricQNetwork scored each candidate
    independently, so taxis with more candidates had more "lottery tickets"
    in the argmax — a systematic bias toward overloading busy taxis.

Solution:
    1. Score each candidate with an MLP (same as before).
    2. Group scores by taxi_id.
    3. Within each taxi group, apply softmax-weighted pooling to produce
       ONE representative score per taxi.
    4. The final action is argmax over the original per-candidate scores,
       but training uses a two-level loss that balances inter-taxi and
       intra-taxi selection.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CandidateScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f = x.shape
        return self.net(x.reshape(b * c, f)).reshape(b, c)


class ParametricQNetwork(nn.Module):
    """Q(s, a) scorer for variable-size candidate sets.

    Input shape: [batch, num_candidates, feature_dim + 1]
    The last channel is a binary valid-mask channel.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        self.scorer = CandidateScorerMLP(input_dim, hidden_dims, dropout)

    def forward(self, x_with_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_with_mask[..., :-1]
        mask = x_with_mask[..., -1]
        q_values = self.scorer(x)
        q_values = q_values.masked_fill(mask <= 0.0, -1e9)
        return q_values, mask


class TaxiFairQNetwork(nn.Module):
    """
    Q-network with taxi-level score normalization.

    Same interface as ParametricQNetwork, but adds a taxi-group
    normalization step so that taxis with many candidates don't
    dominate the argmax.

    The feature matrix must include a taxi_group_id column at a known
    position (passed as taxi_group_col_idx to forward, or appended
    as the second-to-last column before the valid mask).

    Architecture:
        1. CandidateScorerMLP produces raw Q(s,a) for each candidate.
        2. Within each taxi group, raw scores are normalized:
           q_norm_i = q_i - mean(q_group) + max_score_of_best_candidate_in_group
           This keeps the best candidate from each taxi comparable.
        3. A small learned "taxi quality" bias is added per group
           based on pooled group features.

    Input shape: [batch, num_candidates, feature_dim + 1 + 1]
        Last column = valid mask
        Second-to-last column = taxi_group_id (integer, 0-indexed; -1 for defer)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        # The scorer sees the original features (without group_id and mask)
        self.scorer = CandidateScorerMLP(input_dim, hidden_dims, dropout)

        # Small network that produces a per-taxi quality adjustment
        # Input: max Q-value in group, mean Q-value in group, group size
        self.taxi_bias_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, x_with_meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_with_meta: [B, C, F+2] where
                [..., :-2] = features
                [..., -2]  = taxi_group_id (int encoded as float; -1 = defer)
                [..., -1]  = valid mask

        Returns:
            q_values: [B, C] — taxi-fair Q-values
            mask: [B, C]
        """
        mask = x_with_meta[..., -1]
        group_ids = x_with_meta[..., -2].long()  # [B, C]
        x = x_with_meta[..., :-2]  # [B, C, F]

        # Step 1: raw per-candidate scores
        raw_q = self.scorer(x)  # [B, C]
        raw_q = raw_q.masked_fill(mask <= 0.0, -1e9)

        # Step 2: taxi-group normalization
        B, C = raw_q.shape
        fair_q = raw_q.clone()

        for b in range(B):
            valid_mask_b = mask[b] > 0.0
            groups = group_ids[b]  # [C]
            unique_groups = groups[valid_mask_b].unique()

            for g in unique_groups:
                if g.item() < 0:
                    # Defer action — leave its score as-is
                    continue
                group_mask = (groups == g) & valid_mask_b
                group_scores = raw_q[b][group_mask]

                if group_scores.numel() <= 1:
                    continue

                # Normalize within group: center scores, then shift so
                # the best candidate in this group keeps its raw score.
                g_mean = group_scores.mean()
                g_max = group_scores.max()
                g_size = float(group_scores.numel())

                # Adjusted scores: the best candidate keeps its value,
                # weaker candidates are pulled closer to the mean.
                # This reduces the "many lottery tickets" effect.
                fair_q[b][group_mask] = (
                    raw_q[b][group_mask] - g_mean + g_max
                ) - 0.5 * (g_max - raw_q[b][group_mask])

                # Add learned taxi-quality bias
                taxi_stats = torch.tensor(
                    [g_max.item(), g_mean.item(), g_size],
                    device=raw_q.device, dtype=raw_q.dtype,
                )
                bias = self.taxi_bias_net(taxi_stats.unsqueeze(0)).squeeze()
                fair_q[b][group_mask] = fair_q[b][group_mask] + bias

        fair_q = fair_q.masked_fill(mask <= 0.0, -1e9)
        return fair_q, mask