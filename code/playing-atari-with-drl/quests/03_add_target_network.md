# Target network (2013 vs Nature)

**Difficulty:** easy  
**Concept:** frozen **target network** $\hat Q$ for Bellman targets.

- **2013 DQN:** Algorithm 1 bootstraps with the same $Q$ being trained (no separate frozen net).
- **Nature 2015:** every $C$ steps, copy online weights into $\hat Q$ and use $\hat Q$ for targets — slows the feedback loop that causes oscillations.

This lab mirrors that history: CartPole baseline has `target_network: false`; Lunar baseline has `true`.

## Why this is interesting

Raising $Q(s,a)$ also raises tomorrow’s target if you bootstrap from the same net. A delayed $\hat Q$ is the Nature paper’s main stability upgrade over the 2013 workshop version.

## Files

- Config: `rl.target_network`, `rl.target_update_steps` ($C$)
- Code: `dqn/agent.py` (already implemented)

## Experiment

| Game | Change | Hypothesis |
|------|--------|------------|
| CartPole | set `target_network: true` | Often little gain on this easy MDP |
| Lunar | set `target_network: false` | Expect worse / less stable learning |
| Lunar | keep target on; try $C$ = 250 vs 1000 vs 5000 | How “stale” should targets be? |

## Run

```bash
python train.py --config configs/quest_target.yaml --seed 42 --overwrite
```

## What to look for

Eval curve smoothness and final mean return. On Lunar, Gate R (≥ 200) is the solved bar.
