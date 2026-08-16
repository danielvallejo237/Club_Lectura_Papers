# Optimizers: RMSProp vs Adam

**Difficulty:** easy  
**Concept:** which SGD variant trains the Q-network.

The 2013 / Nature DQN papers use **RMSProp**. Later deep RL (e.g. Rainbow) often prefers **Adam**, which is typically less sensitive to learning-rate choice.

## Why this is interesting

Same architecture and replay, different optimizer — isolates an engineering choice that became “modern default.” In this lab, Lunar’s paper-like RMSProp recipe underfit; Adam was needed to reach solved (≥ 200). CartPole often works with either.

## Files

- Config only: `optimizer.type`, `optimizer.lr`

## Experiment

| Run | `optimizer.type` | Suggested `lr` |
|-----|------------------|----------------|
| Paper-like | `rmsprop` | `0.00025` |
| Modern | `adam` | `0.0005` (or `0.001` on CartPole) |

Change **one** of type / lr at a time when comparing.

## Run

```bash
python train.py --config configs/quest_optimizer.yaml --seed 42 --overwrite
```

Prefer **Lunar** for a visible gap; CartPole is a fast sanity check.

## What to look for

Speed of rise in `eval_curve.png`, whether learning stalls, sensitivity if you nudge `lr` by 2–5×.
