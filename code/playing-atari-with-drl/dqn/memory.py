"""Uniform experience replay."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity replay memory with uniform random sampling.

    Replay memory breaks correlations between consecutive environment steps
    by sampling past transitions uniformly at random.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._pos = 0
        self._size = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self._pos] = state
        self.next_states[self._pos] = next_state
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self._size
