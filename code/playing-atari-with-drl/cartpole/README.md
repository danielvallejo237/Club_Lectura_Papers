# CartPole (environment + visualization)

Shared wide CartPole physics, GIF recording, and optional local keyboard play.

**Do not modify these files for quests.** Change `dqn/` instead.

| Module | Purpose |
|--------|---------|
| `env.py` | `make_env()` — same physics for train, eval, record, play |
| `record.py` | Save random / untrained / checkpoint episodes as GIF |
| `play.py` | Local pygame play (display required) |

```bash
python -m cartpole.record --mode random --config configs/cartpole_paperlike.yaml
python -m cartpole.play
```

For a browser demo of CartPole controls, see [jeffjar.me/cartpole.html](https://jeffjar.me/cartpole.html).
