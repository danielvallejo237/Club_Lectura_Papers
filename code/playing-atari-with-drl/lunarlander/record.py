"""Record LunarLander episodes as GIF (single episode or NxN grid).

Grid cells are downscaled uniformly; see ``scripts/lab_gif.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from dqn.agent import DQNAgent
from scripts.lab_gif import (
    GRID_CELL_WIDTH,
    GRID_FRAME_STRIDE,
    GRID_FPS,
    prepare_labeled_episodes,
    shrink_frame,
    stream_grid_gif,
)
from lunarlander.env import make_env


def _load_config(path: str | Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).flatten()


def _end_reason(terminated: bool, truncated: bool, score: float) -> str:
    if terminated:
        return "landed" if score >= 100 else "crash"
    if truncated:
        return "timeout"
    return "done"


def record_episode(
    env: gym.Env,
    policy_fn,
    max_steps: int = 1000,
    seed: int | None = None,
    frame_stride: int = 1,
) -> tuple[list[np.ndarray], float, int, str]:
    frames: list[np.ndarray] = []
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass
    else:
        obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if step % frame_stride == 0:
            frame = env.render()
            if frame is not None:
                frames.append(shrink_frame(frame, GRID_CELL_WIDTH))
        if terminated or truncated:
            break
    end = _end_reason(terminated, truncated, total_reward)
    return frames, total_reward, steps, end


def save_frames(frames: list[np.ndarray], path: Path, fps: int = 30) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to save")
    gif_path = path.with_suffix(".gif")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    return gif_path


def _load_agent(
    mode: str, config_path: str | None, checkpoint: str | None, seed: int
) -> tuple[dict, DQNAgent | None]:
    cpu = torch.device("cpu")
    if mode == "random":
        if not config_path:
            raise ValueError("random mode requires --config")
        return _load_config(config_path), None
    if mode == "untrained":
        if not config_path:
            raise ValueError("untrained mode requires --config")
        config = _load_config(config_path)
        _set_seed(seed)
        env_tmp = make_env(config)
        agent = DQNAgent(config, env_tmp.observation_space, env_tmp.action_space, device=cpu)
        env_tmp.close()
        return config, agent
    if mode == "checkpoint":
        if not checkpoint:
            raise ValueError("checkpoint mode requires --checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        env_tmp = make_env(config)
        agent = DQNAgent(config, env_tmp.observation_space, env_tmp.action_space, device=cpu)
        agent.load_checkpoint(checkpoint)
        env_tmp.close()
        return config, agent
    raise ValueError(f"unknown mode: {mode}")


def record(
    mode: str,
    config_path: str | None = None,
    checkpoint: str | None = None,
    seed: int = 42,
    output: str | None = None,
    max_steps: int = 1000,
) -> Path:
    config, agent = _load_agent(mode, config_path, checkpoint, seed)
    env = make_env(config, render_mode="rgb_array")
    n_actions = int(env.action_space.n)

    def policy_fn(obs: np.ndarray) -> int:
        if mode == "random":
            return int(np.random.randint(n_actions))
        return agent.select_greedy_action(_flatten_obs(obs))

    frames, score, steps, end = record_episode(env, policy_fn, max_steps=max_steps, seed=seed)
    env.close()

    if output is None:
        out_dir = Path("outputs") / "recordings"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"lunar_{mode}.gif")

    gif_path = save_frames(frames, Path(output))
    print(f"saved: {gif_path} (seed={seed} score={score:.0f} steps={steps} end={end})")
    return gif_path


def record_grid(
    mode: str,
    grid_size: int = 3,
    config_path: str | None = None,
    checkpoint: str | None = None,
    seed_start: int = 42,
    output: str | None = None,
    max_steps: int = 400,
) -> Path:
    config, agent = _load_agent(mode, config_path, checkpoint, seed_start)
    env_probe = make_env(config, render_mode="rgb_array")
    n_actions = int(env_probe.action_space.n)
    env_probe.close()
    episodes: list[tuple[list[np.ndarray], int, float, int, str]] = []

    for offset in range(grid_size * grid_size):
        episode_seed = seed_start + offset
        env = make_env(config, render_mode="rgb_array")

        def policy_fn(obs: np.ndarray, _agent=agent) -> int:
            if mode == "random":
                return int(np.random.randint(n_actions))
            return _agent.select_greedy_action(_flatten_obs(obs))

        frames, score, steps, end = record_episode(
            env,
            policy_fn,
            max_steps=max_steps,
            seed=episode_seed,
            frame_stride=GRID_FRAME_STRIDE,
        )
        env.close()
        episodes.append((frames, episode_seed, score, steps, end))
        print(
            f"seed={episode_seed} score={score:.0f} steps={steps} end={end} "
            f"frames={len(frames)}"
        )

    scores = [s for _, _, s, _, _ in episodes]
    print(
        f"grid scores: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}, "
        f"min={np.min(scores):.1f}, max={np.max(scores):.1f}"
    )

    if output is None:
        out_dir = Path("outputs") / "recordings"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"lunar_{mode}_grid{grid_size}.gif")

    labeled = prepare_labeled_episodes(episodes)
    del episodes
    gif_path = stream_grid_gif(labeled, grid_size, Path(output), fps=GRID_FPS)
    del labeled
    print(f"saved grid: {gif_path}")
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record LunarLander episode or grid as GIF.")
    parser.add_argument("--mode", choices=["random", "untrained", "checkpoint"], default="checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=0, help="If >0, record NxN grid.")
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()
    if args.grid > 0:
        record_grid(
            mode=args.mode,
            grid_size=args.grid,
            config_path=args.config,
            checkpoint=args.checkpoint,
            seed_start=args.seed_start,
            output=args.output,
            max_steps=min(args.max_steps, 400) if args.max_steps == 1000 else args.max_steps,
        )
    else:
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
