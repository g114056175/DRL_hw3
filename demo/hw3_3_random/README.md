# HW3-3: Lightning DQN (Random Mode)

## Goal
Use the same DQN idea but implement with a PyTorch Lightning-style module and train in random mode.

## Why random mode is harder
In random mode, player/goal/pit/wall positions vary each episode, so state distribution is broader and sparse success is more common.

## Training tips used
- Gradient clipping (`max_norm=1.0`)
- Exponential LR scheduler
- Target network synchronization

## Output
- `loss_lightning.png`

## Result
- Win rate: `0.388` (500 episodes)
