"""DQN agent: ε-greedy actions, replay, and TD updates.

Quests typically flip config (ε, replay, target, loss, optimizer) or enable
``rl.double_dqn``. Optimizer / loss come from YAML.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from dqn.loss import compute_td_loss
from dqn.memory import ReplayBuffer
from dqn.network import build_q_network
from dqn.policy import epsilon_greedy_action, greedy_action


def _pick_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


class DQNAgent:
    """DQN with optional target network for stabilized Bellman targets."""

    def __init__(
        self,
        config: dict,
        obs_space,
        action_space,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.rl = config["rl"]
        self.loss_type = config.get("loss", {}).get("type", "mse")
        self.gamma = float(self.rl["gamma"])
        self.batch_size = int(self.rl["batch_size"])
        self.use_target = bool(self.rl.get("target_network", False))
        self.target_update_steps = int(self.rl.get("target_update_steps", 1000))
        # Double DQN: select a' with online Q, evaluate with target (needs target_network).
        self.double_dqn = bool(self.rl.get("double_dqn", False))
        if self.double_dqn and not self.use_target:
            raise ValueError("rl.double_dqn=true requires rl.target_network=true")

        self.device = device or _pick_device()
        self.n_actions = int(action_space.n)
        obs_dim = int(obs_space.shape[0])

        self.q_network = build_q_network(config, obs_space, action_space).to(self.device)
        self.target_network: nn.Module | None = None
        if self.use_target:
            self.target_network = copy.deepcopy(self.q_network).to(self.device)
            self.target_network.eval()

        opt_cfg = config.get("optimizer", {})
        opt_type = opt_cfg.get("type", "rmsprop").lower()
        lr = float(opt_cfg.get("lr", 0.00025))
        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(int(self.rl["replay_capacity"]), obs_dim)
        self._train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        return epsilon_greedy_action(self.q_network, state, self.n_actions, epsilon, self.device)

    def select_greedy_action(self, state: np.ndarray) -> int:
        return greedy_action(self.q_network, state, self.device)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Sample a minibatch and perform one Q-learning gradient step."""
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.q_network(states)
        with torch.no_grad():
            target_net = self.target_network if self.use_target else self.q_network
            if self.double_dqn:
                # argmax from online Q; value from target network (Double DQN).
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = compute_td_loss(q_values, actions, targets, self.loss_type)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self.use_target and self._train_steps % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save_checkpoint(self, path: str, step: int, extra: dict | None = None) -> None:
        payload = {
            "step": step,
            "config": self.config,
            "q_network": self.q_network.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(payload["q_network"])
        if self.use_target and self.target_network is not None:
            self.target_network.load_state_dict(payload["q_network"])
        return payload
