"""Q-network architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces


class MLPQNetwork(nn.Module):
    """
    Multi-layer perceptron Q-network.

    Output dimension equals the number of discrete actions. Each output is Q(s, a)
    for that action index.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_q_network(config: dict, obs_space: spaces.Space, action_space: spaces.Space) -> nn.Module:
    """Build a Q-network from config and Gymnasium spaces."""
    if not isinstance(action_space, spaces.Discrete):
        raise ValueError(f"expected Discrete action space, got {type(action_space)}")
    obs_dim = int(obs_space.shape[0])
    net_cfg = config.get("network", {})
    return MLPQNetwork(
        obs_dim=obs_dim,
        n_actions=int(action_space.n),
        hidden_sizes=list(net_cfg.get("hidden_sizes", [128, 128])),
        activation=str(net_cfg.get("activation", "relu")),
    )
