# ε-greedy annealing

**Difficulty:** easy  
**Concept:** exploration via an **ε-greedy** behaviour policy (paper nomenclature).

The 2013 DQN paper anneals ε linearly from **1.0 → 0.1** over the first million frames, then holds 0.1. That leftover ε keeps a little exploration even late in training.

## Why this is interesting

Too little exploration early → the agent never sees good landings / recoveries.  
Too greedy too soon → premature commitment to a bad policy.  
Ending at ε = 0 vs 0.1 asks: does late exploration still help?

## Files

- Config: `rl.epsilon_start`, `rl.epsilon_end`, `rl.epsilon_decay_steps`
- Optional code: `dqn/policy.py` (`linear_schedule`)

## Experiment (pick one change)

| Idea | Example |
|------|---------|
| Faster anneal | Halve `epsilon_decay_steps` |
| Paper-like end ε | `epsilon_end: 0.1` |
| Fully greedy late | `epsilon_end: 0.0` |

## Run

Use CartPole for a quick smoke test; **LunarLander** usually shows a clearer effect.

```bash
# copy a baseline YAML, change one ε key, set a new logging.output_dir
python train.py --config configs/quest_epsilon.yaml --seed 42 --overwrite
```

Compare `outputs/.../plots/eval_curve.png` to the matching baseline run.

## What to look for

Does eval return rise earlier, plateau lower, or get noisier? On Lunar, watch crash vs land variance across seeds.
