"""Evaluate a saved DQN checkpoint with a greedy policy.

Entry point: ``python evaluate.py --checkpoint baselines/<game>_seed42/model.pt --episodes 20``

Loads ``config`` embedded in the checkpoint, rebuilds the agent, and runs greedy
episodes. CartPole prints step-count interpretation; LunarLander prints the
Gym solved note (mean ≥ 200).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import torch

from envs import is_lunarlander_config, make_env_from_config
from dqn.agent import DQNAgent


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_env(env, seed: int) -> None:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    try:
        env.observation_space.seed(seed)
    except AttributeError:
        pass


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).flatten()


def _episode_end_reason(
    config: dict,
    terminated: bool,
    truncated: bool,
    no_time_limit: bool,
    episode_return: float = 0.0,
) -> str:
    if terminated:
        if is_lunarlander_config(config):
            # Gymnasium ends both crash and successful landing as terminated.
            return "landed" if episode_return >= 100.0 else "crash"
        return "fell"
    if truncated and no_time_limit:
        return "cap"
    if truncated:
        return "time_limit"
    return "unknown"


def evaluate(
    checkpoint: str,
    n_episodes: int = 10,
    seed: int = 42,
    save_csv: str | None = None,
    no_time_limit: bool = False,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run greedy episodes; return mean/std/min/max return."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]

    _set_seed(seed)
    if no_time_limit:
        env = make_env_from_config(config, disable_time_limit=True)
        step_cap = max_steps if max_steps is not None else 50000
    else:
        env = make_env_from_config(config)
        step_cap = None

    _seed_env(env, seed)

    agent = DQNAgent(config, env.observation_space, env.action_space)
    agent.load_checkpoint(checkpoint)

    returns: list[float] = []
    episode_rows: list[dict] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = _flatten_obs(obs)
        total = 0.0
        steps = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            if step_cap is not None and steps >= step_cap:
                truncated = True
                break
            action = agent.select_greedy_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = _flatten_obs(obs)
            total += reward
            steps += 1
        end_reason = _episode_end_reason(config, terminated, truncated, no_time_limit, total)
        returns.append(total)
        episode_rows.append({"episode": ep, "return": total, "steps": steps, "end_reason": end_reason})
        print(f"episode {ep}: return={total:.0f} steps={steps} end={end_reason}")

    env.close()
    arr = np.asarray(returns, dtype=np.float32)
    stats = {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
    }
    print(f"episodes: {n_episodes}")
    print(f"mean_return: {stats['mean_return']:.2f}")
    print(f"std_return:  {stats['std_return']:.2f}")
    print(f"min_return:  {stats['min_return']:.2f}")
    print(f"max_return:  {stats['max_return']:.2f}")
    if is_lunarlander_config(config):
        solved = stats["mean_return"] >= 200.0
        print(f"(LunarLander solved threshold: mean_return >= 200; {'solved' if solved else 'not solved'})")
    else:
        print(f"(CartPole score = steps survived; ~{stats['mean_return'] * 0.02:.1f}s at 50 Hz)")
    if no_time_limit:
        print(f"mode: no_time_limit (safety cap per episode: {step_cap})")

    if save_csv:
        pd.DataFrame(episode_rows).to_csv(save_csv, index=False)
        print(f"saved: {save_csv}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy evaluation of a DQN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--no-time-limit", action="store_true", help="CartPole: remove 500-step cap")
    parser.add_argument("--max-steps", type=int, default=None, help="safety cap when --no-time-limit")
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        n_episodes=args.episodes,
        seed=args.seed,
        save_csv=args.save_csv,
        no_time_limit=args.no_time_limit,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
