# Pixel CartPole + CNN

**Difficulty:** hard  
**Concept:** image observations and convolutional Q-network (closer to the Atari paper setup).

## Scope

Not included in the baseline. Requires:

- `render_mode="rgb_array"` frame capture
- preprocessing (resize, grayscale, frame stack)
- CNN Q-network replacing `MLPQNetwork`

## Starting points

- `dqn/network.py` — add `CNNQNetwork`
- `train.py` — image observation pipeline
- Quest notebook for prototyping preprocessing

## Expected effect

Higher compute and implementation complexity; closer to pixel-based DQN described in the source paper.
