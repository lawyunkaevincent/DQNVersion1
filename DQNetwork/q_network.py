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
