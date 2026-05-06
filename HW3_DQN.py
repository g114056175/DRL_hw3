import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from gridworld import Gridworld


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_set = ["u", "d", "l", "r"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_state(env):
    state = env.board.render_np().reshape(1, -1).astype(np.float32)
    return torch.from_numpy(state)


def epsilon_by_step(step, eps_start=1.0, eps_end=0.05, eps_decay=500):
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / eps_decay)


def is_done(reward, step, max_steps):
    return reward != 0 or step >= max_steps


class DQN(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64), num_actions=4):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64), num_actions=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dims[1], 1)
        self.advantage = nn.Linear(hidden_dims[1], num_actions)

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    env_mode,
    q_net,
    episodes=500,
    max_steps=50,
    gamma=0.9,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=500,
    buffer=None,
    batch_size=64,
    target_net=None,
    sync_freq=100,
    use_double=False,
):
    q_net.to(device)
    if target_net is not None:
        target_net.to(device)
        target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    wins = []
    global_step = 0

    for _ in range(episodes):
        env = Gridworld(size=4, mode=env_mode)
        state = get_state(env).to(device)

        for step in range(max_steps):
            global_step += 1
            eps = epsilon_by_step(global_step, eps_start, eps_end, eps_decay)
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(q_net(state).argmax(dim=1).item())

            env.makeMove(action_set[action])
            next_state = get_state(env).to(device)
            reward = env.reward()
            done = is_done(reward, step + 1, max_steps)

            if buffer is not None:
                buffer.push(state, action, reward, next_state, float(done))
                if len(buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    dones = dones.to(device)

                    q_values = q_net(states).gather(1, actions)

                    with torch.no_grad():
                        if target_net is not None:
                            if use_double:
                                next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                                next_q = target_net(next_states).gather(1, next_actions)
                            else:
                                next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                        else:
                            next_q = q_net(next_states).max(dim=1, keepdim=True)[0]

                        targets = rewards + gamma * (1.0 - dones) * next_q

                    loss = loss_fn(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
            else:
                with torch.no_grad():
                    next_q = q_net(next_state).max(dim=1, keepdim=True)[0]
                    target = reward + gamma * (1.0 - float(done)) * float(next_q.item())
                q_val = q_net(state)[0, action]
                loss = loss_fn(q_val, torch.tensor(target, device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if target_net is not None and buffer is not None and global_step % sync_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            state = next_state
            if done:
                break

        wins.append(1 if reward > 0 else 0)

    return losses, wins


def train_keras_dqn_random(
    episodes=400,
    max_steps=50,
    gamma=0.9,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=600,
    batch_size=64,
    sync_freq=200,
):
    import tensorflow as tf
    from tensorflow import keras

    def build_keras_dqn(input_dim=64, num_actions=4):
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(128, activation="relu")(inputs)
        x = keras.layers.Dense(64, activation="relu")(x)
        outputs = keras.layers.Dense(num_actions)(x)
        return keras.Model(inputs, outputs)

    class KerasReplayBuffer:
        def __init__(self, capacity=5000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (
                np.concatenate(states, axis=0).astype(np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.concatenate(next_states, axis=0).astype(np.float32),
                np.array(dones, dtype=np.float32),
            )

        def __len__(self):
            return len(self.buffer)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.95,
        staircase=True,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    q_net = build_keras_dqn()
    target_net = build_keras_dqn()
    target_net.set_weights(q_net.get_weights())

    buffer = KerasReplayBuffer(capacity=4000)
    mse = keras.losses.MeanSquaredError()

    losses = []
    wins = []
    global_step = 0

    for _ in range(episodes):
        env = Gridworld(size=4, mode="random")
        state = env.board.render_np().reshape(1, -1).astype(np.float32)

        for step in range(max_steps):
            global_step += 1
            eps = epsilon_by_step(global_step, eps_start, eps_end, eps_decay)
            if random.random() < eps:
                action = random.randrange(4)
            else:
                q_vals = q_net(state, training=False).numpy()
                action = int(np.argmax(q_vals))

            env.makeMove(action_set[action])
            next_state = env.board.render_np().reshape(1, -1).astype(np.float32)
            reward = env.reward()
            done = is_done(reward, step + 1, max_steps)

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                with tf.GradientTape() as tape:
                    q_vals = q_net(states, training=True)
                    action_masks = tf.one_hot(actions, 4)
                    q_selected = tf.reduce_sum(q_vals * action_masks, axis=1, keepdims=True)

                    next_q = target_net(next_states, training=False)
                    next_max = tf.reduce_max(next_q, axis=1, keepdims=True)
                    targets = rewards.reshape(-1, 1) + gamma * (1.0 - dones.reshape(-1, 1)) * next_max

                    loss = mse(targets, q_selected)

                grads = tape.gradient(loss, q_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
                losses.append(float(loss.numpy()))

            if global_step % sync_freq == 0:
                target_net.set_weights(q_net.get_weights())

            if done:
                break

        wins.append(1 if reward > 0 else 0)

    return q_net, losses, wins


def running_mean(values, window=10):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_curve(path, values, title, ylabel):
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Step")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_demo(output_dir: Path):
    ensure_dir(output_dir)
    results = {}

    naive_net = DQN()
    losses_naive, wins_naive = train_dqn("static", naive_net, episodes=80, max_steps=30)
    results["hw3_1_naive"] = {
        "win_rate": float(sum(wins_naive) / len(wins_naive)),
        "episodes": len(wins_naive),
    }

    replay_net = DQN()
    replay_target = DQN()
    replay_buffer = ReplayBuffer(capacity=1000)
    losses_replay, wins_replay = train_dqn(
        "static",
        replay_net,
        episodes=120,
        max_steps=30,
        buffer=replay_buffer,
        batch_size=64,
        target_net=replay_target,
        sync_freq=100,
    )
    results["hw3_1_replay"] = {
        "win_rate": float(sum(wins_replay) / len(wins_replay)),
        "episodes": len(wins_replay),
    }

    save_curve(output_dir / "hw3_1_losses_naive.png", running_mean(losses_naive, 10), "HW3-1 Naive Loss", "Loss")
    save_curve(output_dir / "hw3_1_losses_replay.png", running_mean(losses_replay, 10), "HW3-1 Replay Loss", "Loss")

    double_net = DQN()
    double_target = DQN()
    double_buffer = ReplayBuffer(capacity=2000)
    losses_double, wins_double = train_dqn(
        "player",
        double_net,
        episodes=120,
        max_steps=40,
        buffer=double_buffer,
        batch_size=64,
        target_net=double_target,
        sync_freq=200,
        use_double=True,
    )
    results["hw3_2_double"] = {
        "win_rate": float(sum(wins_double) / len(wins_double)),
        "episodes": len(wins_double),
    }

    dueling_net = DuelingDQN()
    dueling_target = DuelingDQN()
    dueling_buffer = ReplayBuffer(capacity=2000)
    losses_dueling, wins_dueling = train_dqn(
        "player",
        dueling_net,
        episodes=120,
        max_steps=40,
        buffer=dueling_buffer,
        batch_size=64,
        target_net=dueling_target,
        sync_freq=200,
    )
    results["hw3_2_dueling"] = {
        "win_rate": float(sum(wins_dueling) / len(wins_dueling)),
        "episodes": len(wins_dueling),
    }

    save_curve(output_dir / "hw3_2_losses_double.png", running_mean(losses_double, 10), "HW3-2 Double DQN Loss", "Loss")
    save_curve(output_dir / "hw3_2_losses_dueling.png", running_mean(losses_dueling, 10), "HW3-2 Dueling DQN Loss", "Loss")

    try:
        _, keras_losses, keras_wins = train_keras_dqn_random(episodes=120, max_steps=40)
        results["hw3_3_keras"] = {
            "win_rate": float(sum(keras_wins) / len(keras_wins)),
            "episodes": len(keras_wins),
        }
        save_curve(output_dir / "hw3_3_losses_keras.png", running_mean(keras_losses, 10), "HW3-3 Keras Loss", "Loss")
    except ModuleNotFoundError:
        results["hw3_3_keras"] = {
            "win_rate": None,
            "episodes": 0,
            "note": "TensorFlow not installed",
        }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW3 DQN demo runner")
    parser.add_argument("--demo", action="store_true", help="run demo and save charts")
    parser.add_argument("--out", type=str, default="demo", help="output directory")
    args = parser.parse_args()

    set_seed(42)
    if args.demo:
        metrics = run_demo(Path(args.out))
        print(json.dumps(metrics, indent=2))
    else:
        print("Use --demo to run a short demo and save results.")
