# LunarLander (plumbing)

Env factory, GIF recording, and local play. Training uses `train.py` + `configs/lunarlander_baseline.yaml`.

```bash
python -m lunarlander.record --mode checkpoint --checkpoint baselines/lunarlander_seed42/model.pt --grid 3
python -m lunarlander.play
```

Keys: a / Left, w / Up (main), d / Right; release = coast.  
Browser: [moonlander.seb.ly](http://moonlander.seb.ly/)
