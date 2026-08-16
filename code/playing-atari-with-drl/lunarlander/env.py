"""LunarLander environment factory."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


DEFAULT_LUNARLANDER: dict[str, Any] = {
    "max_episode_steps": 1000,
}


def lunar_env_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge config['env'] over defaults."""
    env_cfg = dict(DEFAULT_LUNARLANDER)
    env_cfg.update(config.get("env") or {})
    return env_cfg


def _strip_time_limit(env: gym.Env) -> gym.Env:
    if isinstance(env, TimeLimit):
        return env.env
    return env


def make_env(
    config: dict[str, Any],
    *,
    render_mode: str | None = None,
    disable_time_limit: bool = False,
) -> gym.Env:
    """
    Create LunarLander-v3 from config.

    Training keeps max_episode_steps (default 1000). Eval can disable the cap.
    """
    env_id = config.get("env_id", "LunarLander-v3")
    env_cfg = lunar_env_from_config(config)
    max_steps = int(env_cfg.get("max_episode_steps", 1000))

    if render_mode is None:
        env = gym.make(env_id, max_episode_steps=max_steps)
    else:
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)

    if disable_time_limit:
        env = _strip_time_limit(env)
    return env
