# CartPole paperlike baseline — results

Same as `baselines/cartpole_seed42/RESULTS.md`. Use this path from notebooks.

**Checkpoint:** `checkpoints/cartpole_baseline/model.pt`  
**Seed:** 42 · **Config:** `configs/cartpole_paperlike.yaml`

## Commands

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
python evaluate.py --checkpoint checkpoints/cartpole_baseline/model.pt --episodes 20 --seed 42
python -m cartpole.record --mode checkpoint --checkpoint checkpoints/cartpole_baseline/model.pt
```

Gate T: mean_return 500.00 (20 ep, timed). Gate L: 50000 steps × 3 ep without fall.
