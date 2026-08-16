"""Epsilon-greedy action selection and exploration schedules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linear interpolation from start to end over decay_steps, then hold at end."""
    if decay_steps <= 0:
        return end
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)


def epsilon_greedy_action(
    q_network: nn.Module,
    state: np.ndarray,
    n_actions: int,
    epsilon: float,
    device: torch.device,
) -> int:
    """With probability epsilon pick a random action; otherwise argmax Q(s, a)."""
    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(state_t)
        return int(q_values.argmax(dim=1).item())


def greedy_action(q_network: nn.Module, state: np.ndarray, device: torch.device) -> int:
    """Select argmax_a Q(s, a)."""
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(state_t)
        return int(q_values.argmax(dim=1).item())
