# CartPole paperlike baseline — results

## Config

- **Config file:** `configs/cartpole_paperlike.yaml`
- **Resolved run config:** `baselines/cartpole_seed42/config.yaml`
- **Seed:** 42
- **Checkpoint:** `baselines/cartpole_seed42/model.pt`

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `total_steps` | 50000 |
| Network | MLP `[128, 128]`, ReLU |
| Optimizer | RMSprop, lr `0.00025` |
| Loss | MSE |
| `gamma` | 0.99 |
| `batch_size` | 64 |
| `replay_capacity` | 50000 |
| `learning_starts` | 1000 |
| `train_frequency` | 1 |
| `epsilon_start` / `epsilon_end` | 1.0 → 0.05 |
| `epsilon_decay_steps` | 30000 |
| `target_network` | false |

## Environment (`env:`)

```yaml
angle_limit_deg: 80.0
x_limit: 8.0
pole_half_length: 1.2
force_mag: 5.0
gravity: 7.0
max_episode_steps: 500
screen_width: 1000
screen_height: 520
```

Training uses TimeLimit 500. Long eval and local play remove the step cap. Agent actions: Left (0) / Right (1).

## Evaluation

### Gate T (timed, 20 episodes, TimeLimit 500)

| metric | value |
|--------|-------|
| mean_return | 500.00 |
| std_return | 0.00 |
| min_return | 500.00 |
| max_return | 500.00 |

### Gate L (untimed, 3 episodes, cap 50000 steps)

| episode | steps | end |
|---------|-------|-----|
| 0 | 50000 | cap (no fall) |
| 1 | 50000 | cap (no fall) |
| 2 | 50000 | cap (no fall) |

## Retrain

```bash
cd code/playing-atari-with-drl
source .venv/bin/activate
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

## Evaluate / record

```bash
python evaluate.py --checkpoint baselines/cartpole_seed42/model.pt --episodes 20 --seed 42
python evaluate.py --checkpoint baselines/cartpole_seed42/model.pt --episodes 3 --seed 123 --no-time-limit --max-steps 50000
python -m cartpole.record --mode checkpoint --checkpoint baselines/cartpole_seed42/model.pt --max-steps 1000
```
