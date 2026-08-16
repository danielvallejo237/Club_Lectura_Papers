"""Record CartPole episodes as GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

from cartpole.env import make_env
from dqn.agent import DQNAgent


def _load_config(path: str | Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_env(env: gym.Env, seed: int) -> None:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    try:
        env.observation_space.seed(seed)
    except AttributeError:
        pass


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).flatten()


def record_episode(env: gym.Env, policy_fn, max_steps: int = 1000) -> list[np.ndarray]:
    """Run one episode and collect RGB frames."""
    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    return frames


def save_frames(frames: list[np.ndarray], path: Path, fps: int = 30) -> Path:
    """Save frames as GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to save")
    gif_path = path.with_suffix(".gif")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    return gif_path


def record(
    mode: str,
    config_path: str | None = None,
    checkpoint: str | None = None,
    seed: int = 42,
    output: str | None = None,
    max_steps: int = 1000,
) -> Path:
    if mode == "random":
        if not config_path:
            raise ValueError("random mode requires --config")
        config = _load_config(config_path)
        agent = None
    elif mode == "untrained":
        if not config_path:
            raise ValueError("untrained mode requires --config")
        config = _load_config(config_path)
        _set_seed(seed)
        env_tmp = make_env(config)
        agent = DQNAgent(config, env_tmp.observation_space, env_tmp.action_space)
        env_tmp.close()
    elif mode == "checkpoint":
        if not checkpoint:
            raise ValueError("checkpoint mode requires --checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        env_tmp = make_env(config)
        agent = DQNAgent(config, env_tmp.observation_space, env_tmp.action_space)
        agent.load_checkpoint(checkpoint)
        env_tmp.close()
    else:
        raise ValueError(f"unknown mode: {mode}")

    env = make_env(config, render_mode="rgb_array")
    if mode != "checkpoint":
        _seed_env(env, seed)

    n_actions = env.action_space.n
    env_id = config["env_id"]

    def policy_fn(obs: np.ndarray) -> int:
        if mode == "random":
            return int(np.random.randint(n_actions))
        state = _flatten_obs(obs)
        return agent.select_greedy_action(state)

    frames = record_episode(env, policy_fn, max_steps=max_steps)
    env.close()

    if output is None:
        out_dir = Path("outputs") / "recordings"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"{mode}_{env_id}.gif")

    gif_path = save_frames(frames, Path(output))
    print(f"saved: {gif_path}")
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record CartPole episode as GIF.")
    parser.add_argument("--mode", choices=["random", "untrained", "checkpoint"], default="checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    record(
        mode=args.mode,
        config_path=args.config,
        checkpoint=args.checkpoint,
        seed=args.seed,
        output=args.output,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
