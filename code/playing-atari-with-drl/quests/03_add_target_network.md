# Add target network

**Difficulty:** medium  
**Concept:** fixed or slowly updated target for Bellman bootstrap.

## Files

- `dqn/agent.py`
- `configs/cartpole_paperlike.yaml`

## Setup

Set in config:

```yaml
rl:
  target_network: true
  target_update_steps: 1000
```

## Run

```bash
python train.py --config configs/cartpole_paperlike.yaml --seed 42 --overwrite
```

## Expected effect

Target network can reduce TD target churn; may improve stability with larger learning rates or harder environments.
