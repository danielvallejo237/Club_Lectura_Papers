# Change network architecture

**Difficulty:** easy–medium  
**Concept:** function approximation capacity and activation.

## Files

- `dqn/network.py`
- `configs/cartpole_paperlike.yaml` (`network.hidden_sizes`, `network.activation`)

## Experiments

- `hidden_sizes: [64]` (single layer)
- `hidden_sizes: [128, 128]` (baseline)
- `hidden_sizes: [256, 256]`
- `activation: tanh` vs `relu`

## Run

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

## Expected effect

Too small → underfitting; too large → slower training, possible instability without target network.
