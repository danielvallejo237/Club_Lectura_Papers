# Change epsilon schedule

**Difficulty:** easy–medium  
**Concept:** exploration rate during training.

## Files

- `dqn/policy.py`
- `configs/cartpole_paperlike.yaml`
- `notebooks/03_quest.ipynb`

## Experiments

| Mode | Change |
|------|--------|
| Easy | Set `rl.epsilon_decay_steps` (e.g. 10000 vs 30000) |
| Medium | Define a custom schedule in a notebook and pass epsilon manually |
| Hard | Add a new schedule function in `dqn/policy.py` |

## Run

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

Compare `plots/eval_curve.png` and `evaluations.csv` across runs.

## Expected effect

Faster epsilon decay → less exploration early → may converge faster or fail if too greedy too soon.
