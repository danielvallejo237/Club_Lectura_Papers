# Try LunarLander

**Difficulty:** medium  
**Concept:** same DQN stack on a harder discrete-control environment.

## Files

- `configs/lunarlander_paperlike.yaml`

## Requirements

```bash
pip install "gymnasium[box2d]"
```

## Run

```bash
python train.py --config configs/lunarlander_paperlike.yaml --seed 42 --overwrite
```

## Expected effect

Longer training horizon, sparser rewards, more variance in eval curves than CartPole.
