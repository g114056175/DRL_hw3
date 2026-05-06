import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buf = deque(maxlen=n_step)

    def push(self, transition):
        self.buf.append(transition)

    def ready(self):
        return len(self.buf) == self.n_step

    def pop(self):
        state, action, _, _, _ = self.buf[0]
        reward, next_state, done = 0.0, self.buf[-1][3], self.buf[-1][4]
        for i, (_, _, r, ns, d) in enumerate(self.buf):
            reward += (self.gamma ** i) * float(r)
            next_state = ns
            if d:
                done = d
                break
        return state, action, reward, next_state, float(done)

    def flush(self):
        items = []
        while self.buf:
            items.append(self.pop())
            self.buf.popleft()
        return items


class PrioritizedReplayBuffer:
    def __init__(self, capacity=6000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        td = np.abs(td_errors.detach().cpu().numpy()).flatten() + 1e-6
        for idx, prio in zip(indices, td):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class LightningDQNModule(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=0.95):
        super().__init__()
        self.model = DQN()
        self.gamma = gamma
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        return [optimizer], [scheduler]


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

    losses, wins = [], []
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
                    states, actions = states.to(device), actions.to(device)
                    rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

                    q_values = q_net(states).gather(1, actions)
                    with torch.no_grad():
                        if target_net is not None and use_double:
                            next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                            next_q = target_net(next_states).gather(1, next_actions)
                        elif target_net is not None:
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


def train_lightning_dqn_random(
    episodes=500,
    max_steps=50,
    gamma=0.95,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=1200,
    batch_size=64,
    sync_freq=200,
):
    module = LightningDQNModule(lr=8e-4, gamma=gamma).to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(module.model.state_dict())
    buffer = ReplayBuffer(capacity=6000)

    optimizer = optim.Adam(module.parameters(), lr=8e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    losses, wins = [], []
    global_step = 0

    for _ in range(episodes):
        env = Gridworld(size=4, mode="random")
        state = get_state(env).to(device)

        for step in range(max_steps):
            global_step += 1
            eps = epsilon_by_step(global_step, eps_start, eps_end, eps_decay)
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(module(state).argmax(dim=1).item())

            env.makeMove(action_set[action])
            next_state = get_state(env).to(device)
            reward = env.reward()
            done = is_done(reward, step + 1, max_steps)
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states, actions = states.to(device), actions.to(device)
                rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

                q_values = module(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                    targets = rewards + gamma * (1.0 - dones) * next_q
                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            if global_step % sync_freq == 0:
                target_net.load_state_dict(module.model.state_dict())
                scheduler.step()

            if done:
                break

        wins.append(1 if reward > 0 else 0)

    return module, losses, wins


def train_rainbow_lite_random(
    episodes=1200,
    max_steps=50,
    gamma=0.95,
    batch_size=64,
    sync_freq=200,
    n_step=3,
):
    q_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)

    per_buffer = PrioritizedReplayBuffer(capacity=8000, alpha=0.6)
    nstep = NStepBuffer(n_step=n_step, gamma=gamma)

    losses, wins = [], []
    global_step = 0

    for ep in range(episodes):
        env = Gridworld(size=4, mode="random")
        state = get_state(env).to(device)

        for step in range(max_steps):
            global_step += 1
            eps = epsilon_by_step(global_step, eps_start=1.0, eps_end=0.02, eps_decay=2600)
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(q_net(state).argmax(dim=1).item())

            env.makeMove(action_set[action])
            next_state = get_state(env).to(device)
            reward = env.reward()
            done = is_done(reward, step + 1, max_steps)

            nstep.push((state, action, reward, next_state, done))
            if nstep.ready():
                per_buffer.push(*nstep.pop())
                nstep.buf.popleft()

            state = next_state

            beta = min(1.0, 0.4 + 0.6 * (ep / max(1, episodes - 1)))
            if len(per_buffer) >= batch_size:
                states, actions, rewards, next_states, dones, weights, indices = per_buffer.sample(batch_size, beta=beta)
                states, actions = states.to(device), actions.to(device)
                rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)
                weights = weights.to(device)

                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                    next_q = target_net(next_states).gather(1, next_actions)
                    targets = rewards + (gamma ** n_step) * (1.0 - dones) * next_q

                td_errors = q_values - targets
                loss = (weights * td_errors.pow(2)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()
                per_buffer.update_priorities(indices, td_errors)
                losses.append(float(loss.item()))

            if global_step % sync_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
                scheduler.step()

            if done:
                break

        for t in nstep.flush():
            per_buffer.push(*t)

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


def save_win_rate_bar(path, results):
    labels = ["Naive", "Replay", "Double", "Dueling", "Lightning", "Rainbow-lite"]
    values = [
        results["hw3_1_naive"]["win_rate"],
        results["hw3_1_replay"]["win_rate"],
        results["hw3_2_double"]["win_rate"],
        results["hw3_2_dueling"]["win_rate"],
        results["hw3_3_lightning"]["win_rate"],
        results["hw3_4_rainbow_lite"]["win_rate"],
    ]
    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, values, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#edc948"])
    plt.ylim(0, 1.0)
    plt.ylabel("Win Rate")
    plt.title("HW3 Win-Rate Comparison")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_demo(output_dir: Path):
    ensure_dir(output_dir)
    results = {}

    naive_net = DQN()
    losses_naive, wins_naive = train_dqn("static", naive_net, episodes=80, max_steps=30)
    results["hw3_1_naive"] = {"win_rate": float(sum(wins_naive) / len(wins_naive)), "episodes": len(wins_naive)}

    replay_net, replay_target = DQN(), DQN()
    replay_buffer = ReplayBuffer(capacity=1000)
    losses_replay, wins_replay = train_dqn(
        "static", replay_net, episodes=120, max_steps=30, buffer=replay_buffer, batch_size=64, target_net=replay_target, sync_freq=100
    )
    results["hw3_1_replay"] = {"win_rate": float(sum(wins_replay) / len(wins_replay)), "episodes": len(wins_replay)}
    save_curve(output_dir / "hw3_1_losses_naive.png", running_mean(losses_naive, 10), "HW3-1 Naive Loss", "Loss")
    save_curve(output_dir / "hw3_1_losses_replay.png", running_mean(losses_replay, 10), "HW3-1 Replay Loss", "Loss")

    double_net, double_target = DQN(), DQN()
    double_buffer = ReplayBuffer(capacity=2000)
    losses_double, wins_double = train_dqn(
        "player", double_net, episodes=120, max_steps=40, buffer=double_buffer, batch_size=64, target_net=double_target, sync_freq=200, use_double=True
    )
    results["hw3_2_double"] = {"win_rate": float(sum(wins_double) / len(wins_double)), "episodes": len(wins_double)}

    dueling_net, dueling_target = DuelingDQN(), DuelingDQN()
    dueling_buffer = ReplayBuffer(capacity=2000)
    losses_dueling, wins_dueling = train_dqn(
        "player", dueling_net, episodes=120, max_steps=40, buffer=dueling_buffer, batch_size=64, target_net=dueling_target, sync_freq=200
    )
    results["hw3_2_dueling"] = {"win_rate": float(sum(wins_dueling) / len(wins_dueling)), "episodes": len(wins_dueling)}
    save_curve(output_dir / "hw3_2_losses_double.png", running_mean(losses_double, 10), "HW3-2 Double DQN Loss", "Loss")
    save_curve(output_dir / "hw3_2_losses_dueling.png", running_mean(losses_dueling, 10), "HW3-2 Dueling DQN Loss", "Loss")

    _, lightning_losses, lightning_wins = train_lightning_dqn_random(episodes=500, max_steps=50)
    results["hw3_3_lightning"] = {"win_rate": float(sum(lightning_wins) / len(lightning_wins)), "episodes": len(lightning_wins)}
    save_curve(output_dir / "hw3_3_losses_lightning.png", running_mean(lightning_losses, 10), "HW3-3 Lightning DQN Loss", "Loss")

    _, rainbow_losses, rainbow_wins = train_rainbow_lite_random(episodes=1200, max_steps=50)
    results["hw3_4_rainbow_lite"] = {"win_rate": float(sum(rainbow_wins) / len(rainbow_wins)), "episodes": len(rainbow_wins)}
    save_curve(output_dir / "hw3_4_losses_rainbow_lite.png", running_mean(rainbow_losses, 10), "HW3-4 Rainbow-lite Loss", "Loss")

    save_win_rate_bar(output_dir / "win_rate_comparison.png", results)

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
