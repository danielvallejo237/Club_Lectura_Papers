# Loss: squared TD error vs Huber

**Difficulty:** easy  
**Concept:** how we penalize Bellman / TD error when fitting $Q$.

- **2013 DQN:** minimize squared error $(y - Q(s,a;\theta))^2$ (MSE).
- **Nature 2015:** clip the TD error (equivalent to **Huber** / smooth L1) so large outliers do not dominate gradients.

## Why this is interesting

Q-learning targets move and can be noisy. Squared loss amplifies huge TD errors; Huber behaves like MSE near zero and like absolute error far away — a small, historically grounded robustness knob. No architecture change required.

## Files

- Config: `loss.type: mse` or `loss.type: huber`
- Code: `dqn/loss.py` (already supports both)

## Experiment

Start from a baseline YAML; flip only the loss:

```yaml
loss:
  type: huber   # baseline labs use mse
```

Set a new `logging.output_dir` (e.g. `outputs/quest_huber_lunar`).

## Run

```bash
python train.py --config configs/quest_huber.yaml --seed 42 --overwrite
```

Try CartPole quickly; **Lunar** (larger reward / TD scale) is where Huber often matters more.

## What to look for

Smoother `loss_curve.png`, fewer blow-ups, similar or better eval return. Ask: did robustness help, or did MSE’s stronger gradient on big errors learn faster?
