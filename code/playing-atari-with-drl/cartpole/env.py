"""Canonical CartPole environment for training and play."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

DEFAULT_CARTPOLE_PHYSICS: dict[str, Any] = {
    "angle_limit_deg": 80.0,
    "x_limit": 8.0,
    "pole_half_length": 1.2,
    "force_mag": 5.0,
    "gravity": 7.0,
    "max_episode_steps": 500,
    "screen_width": 1000,
    "screen_height": 520,
}


def cartpole_physics_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge config['env'] over DEFAULT_CARTPOLE_PHYSICS."""
    physics = dict(DEFAULT_CARTPOLE_PHYSICS)
    physics.update(config.get("env") or {})
    return physics


def apply_cartpole_physics(env: gym.Env, physics: dict[str, Any]) -> gym.Env:
    """Apply project CartPole parameters to an unwrapped CartPole env."""
    base = env.unwrapped
    if not hasattr(base, "x_threshold"):
        return env
    base.theta_threshold_radians = math.radians(float(physics["angle_limit_deg"]))
    base.x_threshold = float(physics["x_limit"])
    base.length = float(physics["pole_half_length"])
    base.polemass_length = base.masspole * base.length
    base.force_mag = float(physics["force_mag"])
    base.gravity = float(physics["gravity"])
    if "screen_width" in physics:
        base.screen_width = int(physics["screen_width"])
    if "screen_height" in physics:
        base.screen_height = int(physics["screen_height"])
    return env


def _strip_time_limit(env: gym.Env) -> gym.Env:
    """Remove TimeLimit so the episode ends only on failure."""
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
    Create CartPole from config.

    Training keeps max_episode_steps (default 500). Play can set disable_time_limit=True.
    """
    env_id = config.get("env_id", "CartPole-v1")
    physics = cartpole_physics_from_config(config)
    max_steps = int(physics.get("max_episode_steps", 500))

    if render_mode is None:
        env = gym.make(env_id, max_episode_steps=max_steps)
    else:
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_steps)

    if disable_time_limit:
        env = _strip_time_limit(env)

    if str(env_id).startswith("CartPole"):
        apply_cartpole_physics(env, physics)
    return env
