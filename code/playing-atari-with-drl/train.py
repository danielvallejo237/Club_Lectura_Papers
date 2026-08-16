"""Train DQN from a YAML config.

Entry point: ``python train.py --config configs/<game>_*.yaml --seed 42 --overwrite``

The loop collects transitions, fills replay, runs ε-greedy actions, periodically
evaluates with a greedy policy, and writes CSV metrics + plots under
``outputs/<output_dir>/seed_<N>/``.

Game choice is only the config file (``env_id`` in YAML). ``envs.py`` and
``cartpole/`` / ``lunarlander/`` are wired automatically — no edits needed there.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from envs import is_lunarlander_config, make_env_from_config
from dqn.agent import DQNAgent, _pick_device
from dqn.policy import linear_schedule

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict:
    """Load a lab YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_dir(output_dir: str, seed: int) -> Path:
    """Directory for one training run: ``outputs/<output_dir>/seed_<seed>/``."""
    return ROOT / output_dir / f"seed_{seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int) -> None:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    try:
        env.observation_space.seed(seed)
    except AttributeError:
        pass


class MetricsLogger:
    """Append-only CSV logger for per-episode and eval metrics."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_rows: list[dict] = []
        self._eval_rows: list[dict] = []

    def log_step(
        self, step: int, episode_reward: float, episode_length: int, epsilon: float, loss: float | None = None
    ) -> None:
        self._metrics_rows.append(
            {"step": step, "episode_reward": episode_reward, "episode_length": episode_length, "epsilon": epsilon, "loss": loss}
        )

    def log_eval(self, step: int, mean_return: float, std_return: float, min_return: float, max_return: float) -> None:
        self._eval_rows.append(
            {"step": step, "mean_return": mean_return, "std_return": std_return, "min_return": min_return, "max_return": max_return}
        )

    def flush(self) -> None:
        if self._metrics_rows:
            pd.DataFrame(self._metrics_rows).to_csv(self.run_dir / "metrics.csv", index=False)
        if self._eval_rows:
            pd.DataFrame(self._eval_rows).to_csv(self.run_dir / "evaluations.csv", index=False)


def save_all_plots(run_dir: Path) -> None:
    """Write reward, loss, and eval PNGs from CSVs in ``run_dir``."""
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(run_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["step"], df["episode_reward"], alpha=0.4, label="episode reward")
        if len(df) >= 20:
            ma = df["episode_reward"].rolling(20, min_periods=1).mean()
            ax.plot(df["step"], ma, label="moving avg (20)")
        ax.set_xlabel("environment step")
        ax.set_ylabel("episode return")
        ax.legend()
        ax.set_title("training reward")
        fig.tight_layout()
        fig.savefig(plot_dir / "reward_curve.png", dpi=120)
        plt.close(fig)
        if not df["loss"].isna().all():
            loss_df = df.dropna(subset=["loss"])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(loss_df["step"], loss_df["loss"], alpha=0.7)
            ax.set_xlabel("environment step")
            ax.set_ylabel("TD loss")
            ax.set_title("training loss")
            fig.tight_layout()
            fig.savefig(plot_dir / "loss_curve.png", dpi=120)
            plt.close(fig)
    eval_path = Path(run_dir) / "evaluations.csv"
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["step"], df["mean_return"], marker="o")
        ax.set_xlabel("environment step")
        ax.set_ylabel("mean eval return")
        ax.set_title("evaluation")
        fig.tight_layout()
        fig.savefig(plot_dir / "eval_curve.png", dpi=120)
        plt.close(fig)


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).flatten()


def evaluate_agent(agent: DQNAgent, config: dict, n_episodes: int, seed: int) -> dict[str, float]:
    """Greedy rollouts for mid-training eval (same policy as ``evaluate.py``)."""
    env = make_env_from_config(config)
    seed_env(env, seed)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = _flatten_obs(obs)
        total_reward = 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.select_greedy_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = _flatten_obs(obs)
            total_reward += reward
        returns.append(total_reward)
    env.close()
    arr = np.asarray(returns, dtype=np.float32)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
    }


def train(config_path: str, seed: int | None = None, overwrite: bool = False) -> Path:
    """Run full DQN training; return path to ``outputs/.../seed_<N>/``."""
    config = load_config(config_path)
    if seed is not None:
        config["seed"] = seed
    seed = int(config.get("seed", 42))

    out = run_dir(config["logging"]["output_dir"], seed)
    if out.exists() and not overwrite:
        raise FileExistsError(f"run directory exists: {out} (pass --overwrite to replace)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    save_config(config, out / "config.yaml")

    set_seed(seed)
    env = make_env_from_config(config)
    seed_env(env, seed)

    print(f"device: {_pick_device()}")
    if config.get("env") and not is_lunarlander_config(config):
        print(f"env physics: {config['env']}")
    elif is_lunarlander_config(config):
        print(f"env: LunarLander ({config.get('env_id', 'LunarLander-v3')})")

    agent = DQNAgent(config, env.observation_space, env.action_space)
    agent.save_checkpoint(str(out / "model_initial.pt"), step=0)

    rl = config["rl"]
    total_steps = int(config["total_steps"])
    learning_starts = int(rl["learning_starts"])
    train_frequency = int(rl["train_frequency"])
    eval_cfg = config["eval"]
    eval_every = int(eval_cfg["eval_every_steps"])
    save_every = int(config["logging"]["save_every_steps"])

    logger = MetricsLogger(out)
    obs, _ = env.reset()
    state = _flatten_obs(obs)
    episode_reward = 0.0
    episode_length = 0
    last_loss: float | None = None

    for step in tqdm(range(1, total_steps + 1), desc="train"):
        epsilon = linear_schedule(
            step, float(rl["epsilon_start"]), float(rl["epsilon_end"]), int(rl["epsilon_decay_steps"])
        )
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = _flatten_obs(next_obs)
        done = terminated or truncated

        agent.store_transition(state, action, float(reward), next_state, done)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if len(agent.replay_buffer) >= learning_starts and step % train_frequency == 0:
            last_loss = agent.train_step()

        if done:
            logger.log_step(step, episode_reward, episode_length, epsilon, last_loss)
            obs, _ = env.reset()
            state = _flatten_obs(obs)
            episode_reward = 0.0
            episode_length = 0

        if step % eval_every == 0:
            stats = evaluate_agent(agent, config, int(eval_cfg["eval_episodes"]), seed + step)
            logger.log_eval(step, **stats)
            print(f"eval step {step}: mean_return={stats['mean_return']:.1f}")

        if step % save_every == 0:
            agent.save_checkpoint(str(out / "checkpoints" / f"step_{step}.pt"), step=step)

    agent.save_checkpoint(str(out / "model_final.pt"), step=total_steps)
    logger.flush()
    save_all_plots(out)
    env.close()
    print(f"finished. outputs: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN. Pick game via --config (cartpole_paperlike or lunarlander_baseline)."
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--overwrite", action="store_true", help="replace existing run directory")
    args = parser.parse_args()
    train(args.config, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
