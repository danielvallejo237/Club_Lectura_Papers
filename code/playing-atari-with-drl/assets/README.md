# Grid assets

Four 3×3 GIFs for the README / notebooks (one greedy episode per cell; overlays show seed, score, steps, end).

| File | Game | Policy |
|------|------|--------|
| `cartpole_untrained_grid.gif` | CartPole | fresh Q-net |
| `cartpole_trained_grid.gif` | CartPole | baseline checkpoint |
| `lunarlander_untrained_grid.gif` | LunarLander | fresh Q-net |
| `lunarlander_trained_grid.gif` | LunarLander | baseline checkpoint |

Regenerate (CPU recommended on small VMs):

```bash
export CUDA_VISIBLE_DEVICES=""
python -m cartpole.record --mode untrained --config configs/cartpole_paperlike.yaml --grid 3 --output assets/cartpole_untrained_grid.gif
python -m cartpole.record --mode checkpoint --checkpoint baselines/cartpole_seed42/model.pt --grid 3 --output assets/cartpole_trained_grid.gif
python -m lunarlander.record --mode untrained --config configs/lunarlander_baseline.yaml --grid 3 --output assets/lunarlander_untrained_grid.gif
python -m lunarlander.record --mode checkpoint --checkpoint baselines/lunarlander_seed42/model.pt --grid 3 --output assets/lunarlander_trained_grid.gif
```

Shared GIF helpers live in `scripts/lab_gif.py` (not for quest edits).
