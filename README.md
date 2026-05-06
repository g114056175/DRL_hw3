# HW3 DQN and Variants

This repository implements Homework 3 on 4x4 Gridworld with DQN family methods.

## 1. Setup & Reference
- Base reference: https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master
- Baseline: instructor-updated starter code in this repo.

### Environment Modes
- `static`: fixed player and fixed objects.
- `player`: random player, fixed objects.
- `random`: player and all objects randomized.

## 2. HW3-1 (Static)
- Naive DQN
- DQN + Experience Replay + Target Network

## 3. HW3-2 (Player)
- Double DQN
- Dueling DQN

## 4. HW3-3 (Random)
Converted DQN implementation to **PyTorch Lightning style module** (`LightningDQNModule`) and trained in random mode.

Training tips:
- Gradient clipping (`max_norm=1.0`)
- Learning-rate scheduler (`ExponentialLR`)
- Target network synchronization

## 5. HW3-4 Bonus (Random)
Implemented **Rainbow-lite**:
- Dueling architecture
- Double DQN target selection
- Prioritized Experience Replay (PER)
- n-step return (n=3)
- Gradient clipping + LR scheduler

## Run
```bash
.\\.venv\\Scripts\\python.exe HW3_DQN.py --demo --out demo
```

## Latest Metrics (2026-05-06)
- hw3_1_naive: `0.575` (80)
- hw3_1_replay: `0.7583` (120)
- hw3_2_double: `0.8417` (120)
- hw3_2_dueling: `0.8667` (120)
- hw3_3_lightning: `0.3880` (500)
- hw3_4_rainbow_lite: `0.4450` (1200)

## Artifacts
- `demo/hw3_1_static/`
- `demo/hw3_2_player/`
- `demo/hw3_3_random/`
- `demo/hw3_4_bonus_rainbow/`
- `demo/summary/`
- `demo/conversation.log`
