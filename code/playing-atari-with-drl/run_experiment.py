"""Run training for multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for multiple seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_script = Path(__file__).resolve().parent / "train.py"
    for seed in args.seeds:
        cmd = [sys.executable, str(train_script), "--config", args.config, "--seed", str(seed)]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
