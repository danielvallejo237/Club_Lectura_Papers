"""Route config ``env_id`` to CartPole or LunarLander.

Used by ``train.py`` and ``evaluate.py``. Game-specific factories live under
``cartpole/env.py`` and ``lunarlander/env.py``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from cartpole.env import make_env as make_cartpole_env
from lunarlander.env import make_env as make_lunarlander_env


def make_env_from_config(
    config: dict[str, Any],
    *,
    render_mode: str | None = None,
    disable_time_limit: bool = False,
) -> gym.Env:
    """Build an env from a lab YAML config."""
    env_id = str(config.get("env_id", "CartPole-v1"))
    if env_id.startswith("CartPole"):
        return make_cartpole_env(config, render_mode=render_mode, disable_time_limit=disable_time_limit)
    if env_id.startswith("LunarLander"):
        return make_lunarlander_env(config, render_mode=render_mode, disable_time_limit=disable_time_limit)
    raise ValueError(f"unsupported env_id: {env_id}")


def is_lunarlander_config(config: dict[str, Any]) -> bool:
    """True when config targets LunarLander."""
    return str(config.get("env_id", "")).startswith("LunarLander")
