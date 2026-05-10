# Homework 3 Report: DQN and Its Variants

## 1. Setup & Reference
- Base concept: DRL in Action 的 GridWorld + DQN 教學脈絡
- Implementation baseline: 本專案統一實作於 `HW3_DQN.py`

## 2. HW3-1 Naive DQN (static)
### Requirement
- 基本 DQN
- Experience Replay 概念與理解

### Implementation
- MLP Q-network: `64 -> 150 -> 100 -> 4`
- Naive 設定與 replay/target 設定比較

### Result
- Naive: Win Rate 100.0%, Avg Steps 7.00
- Replay/Target: Win Rate 100.0%, Avg Steps 7.00

### Explanation
- static 地圖固定、狀態空間小，策略容易收斂到最佳解。

## 3. HW3-2 Double / Dueling (player)
### Requirement
- 實作與比較 Double DQN、Dueling DQN
- 說明如何優於 basic DQN

### Implementation
- Basic DQN
- Double DQN: online 選 action + target 評估
- Dueling DQN: 分離 Value 與 Advantage

### Result
- Basic: 100.0%, 4.465
- Double: 100.0%, 4.27
- Dueling: 100.0%, 4.42

### Explanation
- 在 player mode 中三者都可穩定解題。
- Double/Dueling 的主要改進點在估值偏差控制與學習效率，不一定每次都表現在最終勝率差距上。

## 4. HW3-3 Random mode + Training Tips
### Requirement
- random mode 強化訓練
- Lightning/Keras 方向

### Implementation
- random mode 訓練
- gradient clipping + lr scheduler
- 保留 `DQNLightning` 結構（Lightning-style）

### Result
- Lightning-style(random): Win Rate 86.5%, Avg Steps 9.52

## 5. HW3-4 Bonus Rainbow-lite (random)
### What is Rainbow DQN
Rainbow 是把多個 DQN 改良整合的框架（常見含 Double、Dueling、PER、N-step、Distributional、NoisyNet）。

### This project implementation
- Rainbow-lite: Double + Dueling + Prioritized Replay + Target

### Fair comparison in random mode
- Basic(random): 83.0%, 10.305
- Double(random): 81.5%, 12.115
- Dueling(random): 82.5%, 11.75
- Rainbow-lite(random): 88.5%, 9.01

### Conclusion
- 在 random mode 的同場比較下，Rainbow-lite 取得最佳 win rate 與較低步數，顯示整合式改良對高變化環境較有幫助。

## 6. Output files
- `demo/metrics.json`
- `demo/hw3_1_losses.png`
- `demo/hw3_2_losses.png`
- `demo/hw3_3_losses.png`
- `demo/hw3_4_losses.png`
- `demo/random_compare_losses.png`
- `demo/random_compare_rewards.png`
- `demo/reward_comparison.png`
