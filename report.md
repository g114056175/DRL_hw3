# HW3 Understanding Report (Short)

## Requirement Checklist
- Setup & Reference: based on DRL in Action repo and instructor-updated starter baseline.
- HW3-1: Naive DQN and Experience Replay Buffer implementation in `static` mode.
- HW3-2: Double DQN and Dueling DQN implemented and compared in `player` mode.
- HW3-3: DQN converted to Keras in `random` mode with training tips.
- Evidence: charts + metrics in `demo/` and stage subfolders.

## HW3-1: Naive DQN vs Experience Replay
- Naive DQN updates the Q-network online from the most recent transition. It is simple but noisy because consecutive samples are correlated.
- Experience Replay stores transitions and trains on random minibatches, which breaks correlation and improves stability.
- A target network (a delayed copy of the online network) further stabilizes learning by reducing oscillation in bootstrap targets.

## HW3-2: Double DQN and Dueling DQN
- Double DQN reduces overestimation by selecting the next action with the online network and evaluating it with the target network.
- Dueling DQN separates state value and action advantage, which can learn which states are good even if the action does not matter much.
- Both variants are trained with replay and target network, improving stability and sample efficiency compared to a basic DQN.

## HW3-3: Keras DQN with Training Tips (Random Mode)
- The DQN model is reimplemented in Keras using the same MLP structure.
- Training tips added:
  - Gradient clipping (clipnorm) to stabilize updates.
  - Learning rate schedule (ExponentialDecay) to reduce step size over time.
  - Target network syncing to keep targets stable.
- These techniques help avoid divergence and improve convergence on the random grid setting.

## Demo Snapshot (2026-05-06)
- Naive DQN win rate: 0.575
- Replay DQN win rate: 0.7583
- Double DQN win rate: 0.8417
- Dueling DQN win rate: 0.8667
- Keras DQN win rate: 0.3000
