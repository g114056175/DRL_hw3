# HW3 DQN and Variants

This repository implements Homework 3 (DQN and variants) on a 4x4 Gridworld environment.

## 1. Setup & Reference
- Base reference (DRL in Action, English):
  - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master
- Baseline policy:
  - Use instructor-updated starter code in this repo as the implementation baseline.

### Environment Modes
| Mode | Player Position | Goal/Pit/Wall Position | Usage |
|---|---|---|---|
| `static` | Fixed `(0,3)` | Fixed: Goal `(0,0)`, Pit `(0,1)`, Wall `(1,1)` | Correctness check and reproducibility |
| `player` | Random | Fixed: Goal `(0,0)`, Pit `(0,1)`, Wall `(1,1)` | Generalization from different starts |
| `random` | Random | Random | Stronger policy robustness training |

## 2. HW3-1 (Static Mode)
- Naive DQN
- DQN + Experience Replay Buffer (+ Target Network)

Artifacts:
- `demo/hw3_1_static/loss_naive.png`
- `demo/hw3_1_static/loss_replay.png`

## 3. HW3-2 (Player Mode)
Implemented and compared:
- Double DQN
- Dueling DQN

Artifacts:
- `demo/hw3_2_player/loss_double.png`
- `demo/hw3_2_player/loss_dueling.png`

## 4. HW3-3 (Random Mode + Training Tips)
Converted from PyTorch DQN to Keras DQN.

Training stabilization tips used:
- Gradient clipping (`clipnorm=1.0`)
- Learning-rate scheduling (`ExponentialDecay`)
- Target network periodic synchronization

Artifacts:
- `demo/hw3_3_random/loss_keras.png`

## Short Understanding Report
- `report.md`

## Conversation Log
- `demo/conversation.log`

## Run Demo
```bash
.\\.venv\\Scripts\\python.exe HW3_DQN.py --demo --out demo
```

## Latest Metrics (2026-05-06)
- Naive DQN win rate: `0.575` (80 episodes)
- Replay DQN win rate: `0.758` (120 episodes)
- Double DQN win rate: `0.842` (120 episodes)
- Dueling DQN win rate: `0.867` (120 episodes)
- Keras DQN (random mode) win rate: `0.300` (120 episodes)

## Visual Summary
- `demo/summary/win_rate_comparison.png`
- `demo/summary/metrics.json`
