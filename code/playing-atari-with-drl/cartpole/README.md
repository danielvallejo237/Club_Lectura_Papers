# CartPole (plumbing)

Env factory, GIF recording, and local play. Training uses `train.py` + `configs/cartpole_paperlike.yaml`.

```bash
python -m cartpole.record --mode checkpoint --checkpoint baselines/cartpole_seed42/model.pt --grid 3
python -m cartpole.play
```

Browser demo: [jeffjar.me/cartpole.html](https://jeffjar.me/cartpole.html)
