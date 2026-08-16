# Frozen baselines

Reference weights for watch-without-train and regression checks.

| Path | Seed | Config |
|------|------|--------|
| `cartpole_seed42/` | 42 | `configs/cartpole_paperlike.yaml` |

Do not overwrite these files when experimenting. Reproduce with:

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```
