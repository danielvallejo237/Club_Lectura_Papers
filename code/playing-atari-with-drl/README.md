# Playing Atari with Deep RL — CartPole DQN Lab

Paperlike DQN on wide CartPole: train, evaluate, optional quests. Colab-first.

**Paper:** [arXiv:1312.5602](https://arxiv.org/pdf/1312.5602) · local PDF: `playing-atari-with-drl.pdf`

---

## Start here

| Step | Action |
|------|--------|
| 1 | **Feel the game** — [CartPole in the browser](https://jeffjar.me/cartpole.html) (arrow keys / h/l) |
| 2 | **Watch agents** — open `notebooks/01_watch.ipynb` (Colab or local Jupyter) |
| 3 | **Train baseline** — `notebooks/02_train.ipynb` or `python train.py --config configs/cartpole_paperlike.yaml --seed 42` |
| 4 | **Quest** — pick one below, edit `dqn/`, re-run training in `notebooks/03_quest.ipynb` |

**Trained weights (local):** `checkpoints/cartpole_baseline/model.pt` or frozen copy `baselines/cartpole_seed42/model.pt`

---

## Layout

```text
dqn/        ← edit for quests (memory, loss, network, policy, agent)
cartpole/   ← env + GIF + local play — do not change for quests
train.py    ← training loop, metrics CSV, plots
configs/    ← hyperparameters
notebooks/  ← Colab entry points
quests/     ← quest how-tos
```

Human play uses the same `env:` physics as training. The agent uses Left/Right only; local play may coast on key release.

---

## Quests

| # | Goal | Edit | Guide |
|---|------|------|-------|
| 1 | Epsilon schedule | `dqn/policy.py`, config `rl.epsilon_*` | [quests/01_change_epsilon_schedule.md](quests/01_change_epsilon_schedule.md) |
| 2 | Replay memory | `dqn/memory.py`, config `rl.replay_capacity` | [quests/02_change_replay_memory.md](quests/02_change_replay_memory.md) |
| 3 | Target network | `dqn/agent.py`, config `rl.target_network` | [quests/03_add_target_network.md](quests/03_add_target_network.md) |
| 4 | Optimizer | config `optimizer.type` | [quests/04_compare_optimizers.md](quests/04_compare_optimizers.md) |
| 5 | LunarLander | `configs/lunarlander_paperlike.yaml` | [quests/05_try_lunarlander.md](quests/05_try_lunarlander.md) |
| 6 | Network size | `dqn/network.py`, config `network.*` | [quests/06_change_network_architecture.md](quests/06_change_network_architecture.md) |
| 7 | Pixel + CNN | `dqn/network.py`, `train.py` | [quests/07_pixel_cartpole_cnn.md](quests/07_pixel_cartpole_cnn.md) |

After each run: check `outputs/<experiment>/seed_<N>/plots/` and `metrics.csv`.

---

## Commands

```bash
# setup (local)
bash scripts/setup.sh && source .venv/bin/activate

# train baseline (reproduce seed 42)
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite

# evaluate
python evaluate.py --checkpoint checkpoints/cartpole_baseline/model.pt --episodes 20 --seed 42

# record GIF
python -m cartpole.record --mode checkpoint --checkpoint checkpoints/cartpole_baseline/model.pt

# local play (display required)
python -m cartpole.play
```

---

## Colab

```python
REPO_URL = "https://github.com/danielvallejo237/Club_Lectura_Papers.git"
%cd Club_Lectura_Papers/code/playing-atari-with-drl
!pip install -q -r requirements.txt
```

Open `notebooks/01_watch.ipynb`. Save a copy to Drive before editing.

---

## Score

CartPole reward is +1 per timestep. Score = steps survived; time ≈ score × 0.02 s. Training truncates at 500 steps; play and long eval end on fall.
