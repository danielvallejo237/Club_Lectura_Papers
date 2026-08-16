# Optional experiments

See the quest table in the main [README](../README.md#quests).

CartPole trains faster and is usually easier / more stable. LunarLander takes longer and is messier — use it when you want quest differences to stand out.

| # | Guide |
|---|--------|
| 1 | [ε-greedy annealing](01_change_epsilon_schedule.md) |
| 2 | [Experience replay](02_change_replay_memory.md) |
| 3 | [Target network](03_add_target_network.md) |
| 4 | [RMSProp vs Adam](04_compare_optimizers.md) |
| 5 | [MSE vs Huber](05_mse_vs_huber.md) |
| 6 | [Double DQN](06_double_dqn.md) |
| 7 | [Discount $\gamma$](07_discount_gamma.md) |

Copy a config, change one key, set a new `logging.output_dir`, then:

```bash
python train.py --config configs/quest_....yaml --seed 42 --overwrite
```

Quest YAML copies (`configs/quest_*.yaml`) are gitignored.
