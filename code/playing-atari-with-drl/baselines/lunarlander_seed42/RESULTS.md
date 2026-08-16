# LunarLander baseline — results

## Config

- **Config file:** `configs/lunarlander_baseline.yaml`
- **Resolved run config:** `baselines/lunarlander_seed42/config.yaml`
- **Seed:** 42
- **Checkpoint:** `baselines/lunarlander_seed42/model.pt`
- **Training steps:** 300000

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Network | MLP `[256, 256]`, ReLU |
| Optimizer | Adam, lr `0.0005` |
| Loss | MSE |
| `gamma` | 0.99 |
| `batch_size` | 64 |
| `replay_capacity` | 100000 |
| `learning_starts` | 5000 |
| `train_frequency` | 1 |
| `epsilon_start` / `epsilon_end` | 1.0 → 0.01 |
| `epsilon_decay_steps` | 150000 |
| `target_network` | true |
| `target_update_steps` | 1000 |

## vs paper

Atari DQN uses RMSProp + CNN + pixels. An RMSProp MLP 128×128 run at 200k steps reached eval mean ~25 (not solved). Adam + wider net + 300k steps reached Gate R. See comments in `configs/lunarlander_baseline.yaml`.

## Evaluation

### Gate R (greedy, 100 episodes, seed 42)

| metric | value |
|--------|-------|
| mean_return | 207.59 |
| std_return | 184.01 |
| min_return | -480.36 |
| max_return | 308.88 |
| solved (≥200) | yes |

### Training eval (last checkpoint, 20 episodes)

| step | mean_return |
|------|-------------|
| 300000 | 250.1 |

## Retrain

```bash
python train.py --config configs/lunarlander_baseline.yaml --seed 42 --overwrite
```

## Evaluate / record

```bash
python evaluate.py --checkpoint baselines/lunarlander_seed42/model.pt --episodes 100 --seed 42
python -m lunarlander.record --mode checkpoint --checkpoint baselines/lunarlander_seed42/model.pt --grid 3 --output assets/lunarlander_trained_grid.gif
```
