# Compare optimizers

**Difficulty:** easy  
**Concept:** RMSProp vs Adam on the same architecture and replay setup.

## Files

- `configs/cartpole_paperlike.yaml` (`optimizer.type`, `optimizer.lr`)

## Experiments

| Run | `optimizer.type` | `optimizer.lr` |
|-----|------------------|----------------|
| A | rmsprop | 0.00025 |
| B | adam | 0.001 |

## Run

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

## Expected effect

Adam often converges faster on small MLP tasks; RMSProp matches the original DQN paper defaults.
