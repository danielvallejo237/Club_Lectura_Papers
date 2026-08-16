# Double DQN

**Difficulty:** medium  
**Concept:** reduce **overestimation** of action values from $\max_{a'} Q(s',a')$.

Standard DQN (and Nature DQN) picks and evaluates the next action with the same network:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**Double DQN** (van Hasselt et al., 2016) decouples selection and evaluation:

$$y = r + \gamma \, Q\big(s', \arg\max_{a'} Q(s',a';\theta),\, \theta^-\big)$$

Online $Q$ chooses $a'$; the **target network** evaluates it.

## Why this is interesting

$\max$ over noisy value estimates is optimistically biased — the agent can chase phantom high-Q actions. Double DQN is a classic post-2013 fix that still fits this codebase (few lines). Needs a target network (`target_network: true`).

## Files

- Config: `rl.double_dqn: true` (requires `rl.target_network: true`)
- Code: `dqn/agent.py` (flag already wired)

## Experiment

On Lunar (recommended), start from `configs/lunarlander_baseline.yaml` (target already on):

```yaml
rl:
  target_network: true
  double_dqn: true
```

On CartPole, turn target on first, then enable Double DQN.

## Run

```bash
python train.py --config configs/quest_double_dqn.yaml --seed 42 --overwrite
```

## What to look for

More stable eval, less “confident but wrong” crashing on Lunar, possibly lower average predicted Q for the same policy quality. Compare against the same seed without `double_dqn`.
