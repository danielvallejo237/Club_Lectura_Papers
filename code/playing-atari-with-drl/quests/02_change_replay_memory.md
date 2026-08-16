# Experience replay

**Difficulty:** easy–medium  
**Concept:** **experience replay** / replay memory $\mathcal{D}$ — the central 2013 DQN idea.

The paper stores transitions $(s,a,r,s')$ and samples **uniform minibatches** so updates are less correlated and the data distribution is averaged over past behaviour.

## Why this is interesting

Without replay, consecutive frames are highly correlated and the policy’s newest quirks dominate the data — the paper argues this can oscillate or diverge. Capacity also matters: a tiny buffer forgets old behaviours too fast.

The paper itself notes uniform sampling is limited and hints at prioritizing transitions we can learn from most (later: prioritized experience replay).

## Files

- Config: `rl.replay_capacity`, `rl.learning_starts`
- Code: `dqn/memory.py`

## Experiment (pick one)

| Idea | Change |
|------|--------|
| Small memory | `replay_capacity: 1000` (or 5000 on Lunar) |
| Large memory | raise capacity toward paper scale (relative to `total_steps`) |
| Harder | temporarily train from recent consecutive samples only (ablate shuffle) — expect instability |

## Run

```bash
python train.py --config configs/quest_replay.yaml --seed 42 --overwrite
```

CartPole can still solve with a small buffer; **Lunar** is the stress test.

## What to look for

Noisier / collapsing eval curves with a tiny buffer; smoother learning with enough replay. TD loss spikes if correlations return.
