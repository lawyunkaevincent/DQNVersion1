from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    done: bool


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_state_exists: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = int(capacity)
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        states = _pad_state_batch([t.state for t in batch])
        next_states = _pad_state_batch([
            t.next_state if t.next_state is not None else np.zeros((1, batch[0].state.shape[1]), dtype=np.float32)
            for t in batch
        ])
        next_exists = np.asarray([0.0 if t.next_state is None else 1.0 for t in batch], dtype=np.float32)
        return ReplayBatch(
            states=torch.from_numpy(states).to(device),
            actions=torch.tensor([t.action_index for t in batch], dtype=torch.int64, device=device),
            rewards=torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device),
            next_states=torch.from_numpy(next_states).to(device),
            next_state_exists=torch.from_numpy(next_exists).to(device),
            dones=torch.tensor([1.0 if t.done else 0.0 for t in batch], dtype=torch.float32, device=device),
        )


def _pad_state_batch(states: Sequence[np.ndarray]) -> np.ndarray:
    max_cands = max(s.shape[0] for s in states)
    feat_dim = states[0].shape[1]
    out = np.zeros((len(states), max_cands, feat_dim), dtype=np.float32)
    for i, s in enumerate(states):
        out[i, : s.shape[0], :] = s
    return out
