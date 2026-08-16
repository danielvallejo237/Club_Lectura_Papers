# Change replay memory

**Difficulty:** easy  
**Concept:** experience replay capacity and sample diversity.

## Files

- `dqn/memory.py`
- `configs/cartpole_paperlike.yaml`

## Experiments

Try `rl.replay_capacity`: 1000, 10000, 50000. Optional ablation: train without replay (requires code change to skip buffer sampling).

## Run

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

## Expected effect

Smaller buffer → noisier updates; very small buffer may destabilize learning on harder tasks.
