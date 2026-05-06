# HW3 Understanding Report (Short)

## Why HW3-3 looked low before
The previous HW3-3 result used Keras DQN with short training in `random` mode, which is harder than `static` and `player` modes. That produced lower win rate.

## Current HW3-3 implementation
HW3-3 now uses a PyTorch Lightning-style DQN module in random mode with stabilization tips:
- gradient clipping
- LR scheduler
- target network sync

This improves win rate from earlier Keras run (`0.30`) to current `0.388`.

## HW3-4 bonus: Rainbow-lite in random mode
Implemented and tested:
- Double DQN
- Dueling Network
- Prioritized Replay
- n-step return (n=3)

Result: `0.445` win rate, better than current HW3-3 baseline (`0.388`) in the same random mode setting.

## Final snapshot (2026-05-06)
- Naive DQN: 0.575
- Replay DQN: 0.7583
- Double DQN: 0.8417
- Dueling DQN: 0.8667
- Lightning DQN (random): 0.388
- Rainbow-lite (random): 0.445
