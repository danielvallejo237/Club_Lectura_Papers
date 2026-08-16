# Discount factor $\gamma$ (return & credit assignment)

**Difficulty:** easy  
**Concept:** what the agent is actually optimizing — the **discounted return**.

In the paper, the return from time $t$ is

$$R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$$

and the action-value $Q^*(s,a)$ is the expected return after taking $a$ in $s$. The **Bellman** backup uses the same $\gamma$:

$$Q(s,a) \;\approx\; r + \gamma \max_{a'} Q(s',a')$$

(with a terminal mask when the episode ends). Your configs default to $\gamma = 0.99$ (same ballpark as DQN).

## Why this is interesting (RL thinking)

This is not a stability trick like Adam or Huber. Changing $\gamma$ changes the **definition of success**:

| $\gamma$ | Intuition |
|----------|-----------|
| Near **0** | Myopic: mostly cares about immediate $r$. CartPole may still pad steps; Lunar may burn fuel / thrash without planning a soft landing. |
| **0.99** (default) | Long horizon: future crashes and landings matter. |
| **1.0** (or very close) | Almost undiscounted sum of rewards — credit assignment over the whole episode; often noisier / harder to learn. |

Ask yourself: *if the lander gets a big positive reward only at the end, how far back in time should that signal reach?* That *is* $\gamma$.

## Files

- Config only: `rl.gamma`
- The backup already uses it in `dqn/agent.py` (`self.gamma`)

## Experiment

Copy a baseline YAML; change **only** $\gamma$ and `logging.output_dir`.

| Run | `rl.gamma` | Hypothesis |
|-----|------------|------------|
| A | `0.9` | Shorter effective horizon |
| B | `0.99` | Baseline |
| C | `0.999` or `1.0` | Longer credit; watch instability |

Start on **CartPole** (fast), then **LunarLander** (where planning a landing makes the effect clearer).

```bash
python train.py --config configs/quest_gamma.yaml --seed 42 --overwrite
```

## What to look for

- Eval return: does a myopic agent “solve” CartPole by luck but fail Lunar?
- Behaviour in grids / play: impatient thrusters vs smoother approach.
- Optional stretch (paper-style): during eval, log average $\max_a Q(s,a)$ over states — the paper’s Figure 2 tracks this as a smoother learning signal than episode reward alone.

## Takeaway

Hyperparameters like ε and replay change *how* we learn $Q$. **$\gamma$ changes what $Q$ means.** That is the RL core under the DQN machinery.
