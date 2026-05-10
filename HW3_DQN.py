import argparse
import copy
import json
import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gridworld import Gridworld

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except Exception:
    LIGHTNING_AVAILABLE = False

ACTIONS = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_from_game(game: Gridworld, noise_scale: float = 100.0):
    arr = game.board.render_np().reshape(1, 64)
    arr = arr + np.random.rand(1, 64) / noise_scale
    return torch.from_numpy(arr).float()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def push(self, *args):
        self.data.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)


class DQN(nn.Module):
    def __init__(self, in_dim=64, h1=150, h2=100, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=64, h1=150, h2=100, out_dim=4):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.value = nn.Linear(h2, 1)
        self.adv = nn.Linear(h2, out_dim)

    def forward(self, x):
        f = self.feat(x)
        v = self.value(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.data = []
        self.priorities = []
        self.pos = 0

    def push(self, transition: Transition, priority: float = 1.0):
        p = max(priority, 1e-6)
        if len(self.data) < self.capacity:
            self.data.append(transition)
            self.priorities.append(p)
        else:
            self.data[self.pos] = transition
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self.data), batch_size, p=probs)
        samples = [self.data[i] for i in idx]
        weights = (len(self.data) * probs[idx]) ** (-beta)
        weights /= weights.max()
        return samples, idx, torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = float(abs(e) + 1e-6)

    def __len__(self):
        return len(self.data)


@dataclass
class TrainCfg:
    mode: str
    episodes: int = 2000
    mem_size: int = 10000
    batch_size: int = 128
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    lr: float = 1e-3
    max_moves: int = 50
    sync_freq: int = 400
    grad_clip: float = 5.0


def choose_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        q = model(state)
    return int(torch.argmax(q, dim=1).item())


def train_dqn_variant(cfg: TrainCfg, variant='basic', device='cpu'):
    if variant == 'dueling':
        online = DuelingDQN().to(device)
    else:
        online = DQN().to(device)

    use_target = variant in {'double', 'dueling', 'rainbow'}
    target = copy.deepcopy(online).to(device) if use_target else None
    if target is not None:
        for p in target.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.episodes // 2), gamma=0.5)
    loss_fn = nn.MSELoss(reduction='none')
    replay = PrioritizedReplayBuffer(cfg.mem_size) if variant == 'rainbow' else ReplayBuffer(cfg.mem_size)

    epsilon = cfg.epsilon_start
    global_step = 0
    losses, rewards = [], []

    train_started = False
    for ep in range(cfg.episodes):
        game = Gridworld(size=4, mode=cfg.mode)
        state = state_from_game(game).to(device)
        ep_reward = 0.0

        for _ in range(cfg.max_moves):
            global_step += 1
            action = choose_action(online, state, epsilon)
            game.makeMove(ACTIONS[action])
            next_state = state_from_game(game).to(device)
            reward = float(game.reward())
            done = reward != -1.0
            ep_reward += reward

            if variant == 'rainbow':
                replay.push(Transition(state, action, reward, next_state, done), priority=1.0)
            else:
                replay.push(state, action, reward, next_state, done)

            state = next_state

            if len(replay) >= cfg.batch_size:
                if variant == 'rainbow':
                    batch, idx, w = replay.sample(cfg.batch_size, beta=min(1.0, 0.4 + ep / cfg.episodes))
                    w = w.to(device)
                else:
                    batch = replay.sample(cfg.batch_size)
                    w = torch.ones((cfg.batch_size, 1), dtype=torch.float32, device=device)

                b_s = torch.cat([b.state for b in batch]).to(device)
                b_a = torch.tensor([b.action for b in batch], dtype=torch.int64, device=device).view(-1, 1)
                b_r = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device).view(-1, 1)
                b_ns = torch.cat([b.next_state for b in batch]).to(device)
                b_d = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device).view(-1, 1)

                q = online(b_s).gather(1, b_a)
                with torch.no_grad():
                    if target is None:
                        max_next_q = online(b_ns).max(dim=1, keepdim=True)[0]
                    elif variant in {'double', 'dueling', 'rainbow'}:
                        next_actions = online(b_ns).argmax(dim=1, keepdim=True)
                        max_next_q = target(b_ns).gather(1, next_actions)
                    else:
                        max_next_q = target(b_ns).max(dim=1, keepdim=True)[0]
                    y = b_r + cfg.gamma * (1.0 - b_d) * max_next_q

                per_sample_loss = loss_fn(q, y)
                loss = (per_sample_loss * w).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                optimizer.step()
                train_started = True
                losses.append(float(loss.item()))

                if variant == 'rainbow':
                    td = (q.detach() - y.detach()).abs().view(-1).cpu().numpy()
                    replay.update_priorities(idx, td)

                if target is not None and global_step % cfg.sync_freq == 0:
                    target.load_state_dict(online.state_dict())

            if done:
                break

        rewards.append(ep_reward)
        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
        if train_started:
            scheduler.step()

    return online, {'losses': losses, 'rewards': rewards}


def evaluate(model, mode='random', episodes=200, device='cpu'):
    model.eval()
    wins = 0
    steps = []
    with torch.no_grad():
        for _ in range(episodes):
            game = Gridworld(size=4, mode=mode)
            state = state_from_game(game, noise_scale=10.0).to(device)
            for t in range(1, 51):
                action = int(torch.argmax(model(state), dim=1).item())
                game.makeMove(ACTIONS[action])
                state = state_from_game(game, noise_scale=10.0).to(device)
                reward = game.reward()
                if reward != -1:
                    if reward > 0:
                        wins += 1
                    steps.append(t)
                    break
                if t == 50:
                    steps.append(t)
    return {'win_rate': 100.0 * wins / episodes, 'avg_steps': float(np.mean(steps))}


class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros(1)


class DQNLightning(pl.LightningModule if LIGHTNING_AVAILABLE else object):
    def __init__(self, cfg: TrainCfg):
        if LIGHTNING_AVAILABLE:
            super().__init__()
        self.cfg = cfg
        self.model = DQN(in_dim=64, h1=128, h2=256, out_dim=4)
        self.replay = ReplayBuffer(cfg.mem_size)
        self.loss_fn = nn.MSELoss()
        self.epsilon = cfg.epsilon_start

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.cfg.episodes // 2), gamma=0.5)
        return [optimizer], [sched]

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=1)


def rolling(arr, window=50):
    if len(arr) < window:
        return np.array(arr)
    x = np.array(arr)
    return np.convolve(x, np.ones(window) / window, mode='valid')


def save_plot(series_dict, title, out_path, xlabel):
    plt.figure(figsize=(10, 5))
    for name, vals in series_dict.items():
        r = rolling(vals, 50)
        plt.plot(r, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_all(args):
    os.makedirs('demo', exist_ok=True)
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metrics = {}

    m1_naive, h1_naive = train_dqn_variant(TrainCfg(mode='static', episodes=args.ep_hw1), variant='basic', device=device)
    m1_replay, h1_replay = train_dqn_variant(TrainCfg(mode='static', episodes=args.ep_hw1), variant='double', device=device)
    metrics['hw3_1_naive'] = evaluate(m1_naive, mode='static', device=device)
    metrics['hw3_1_replay'] = evaluate(m1_replay, mode='static', device=device)
    save_plot({'Naive': h1_naive['losses'], 'Replay+Target': h1_replay['losses']}, 'HW3-1 Loss (Static)', 'demo/hw3_1_losses.png', 'Train updates')

    m_basic, h_basic = train_dqn_variant(TrainCfg(mode='player', episodes=args.ep_hw2), variant='basic', device=device)
    m_double, h_double = train_dqn_variant(TrainCfg(mode='player', episodes=args.ep_hw2), variant='double', device=device)
    m_duel, h_duel = train_dqn_variant(TrainCfg(mode='player', episodes=args.ep_hw2), variant='dueling', device=device)
    metrics['hw3_2_basic'] = evaluate(m_basic, mode='player', device=device)
    metrics['hw3_2_double'] = evaluate(m_double, mode='player', device=device)
    metrics['hw3_2_dueling'] = evaluate(m_duel, mode='player', device=device)
    save_plot({'Basic': h_basic['losses'], 'Double': h_double['losses'], 'Dueling': h_duel['losses']}, 'HW3-2 Loss (Player)', 'demo/hw3_2_losses.png', 'Train updates')

    m_light, h_light = train_dqn_variant(TrainCfg(mode='random', episodes=args.ep_hw3, lr=5e-4, grad_clip=3.0), variant='double', device=device)
    metrics['hw3_3_lightning_style'] = evaluate(m_light, mode='random', device=device)
    save_plot({'Lightning-style': h_light['losses']}, 'HW3-3 Loss (Random)', 'demo/hw3_3_losses.png', 'Train updates')

    m_rainbow, h_rainbow = train_dqn_variant(TrainCfg(mode='random', episodes=args.ep_hw4, lr=5e-4), variant='rainbow', device=device)
    metrics['hw3_4_rainbow_lite'] = evaluate(m_rainbow, mode='random', device=device)
    save_plot({'Rainbow-lite': h_rainbow['losses']}, 'HW3-4 Loss (Random)', 'demo/hw3_4_losses.png', 'Train updates')

    # Fair comparison on random mode across all variants
    m_r_basic, h_r_basic = train_dqn_variant(TrainCfg(mode='random', episodes=args.ep_cmp), variant='basic', device=device)
    m_r_double, h_r_double = train_dqn_variant(TrainCfg(mode='random', episodes=args.ep_cmp), variant='double', device=device)
    m_r_duel, h_r_duel = train_dqn_variant(TrainCfg(mode='random', episodes=args.ep_cmp), variant='dueling', device=device)
    m_r_rainbow, h_r_rainbow = train_dqn_variant(
        TrainCfg(mode='random', episodes=args.ep_cmp, lr=5e-4), variant='rainbow', device=device
    )

    metrics['random_compare_basic'] = evaluate(m_r_basic, mode='random', device=device)
    metrics['random_compare_double'] = evaluate(m_r_double, mode='random', device=device)
    metrics['random_compare_dueling'] = evaluate(m_r_duel, mode='random', device=device)
    metrics['random_compare_rainbow_lite'] = evaluate(m_r_rainbow, mode='random', device=device)

    save_plot(
        {
            'Basic(random)': h_r_basic['losses'],
            'Double(random)': h_r_double['losses'],
            'Dueling(random)': h_r_duel['losses'],
            'Rainbow-lite(random)': h_r_rainbow['losses'],
        },
        'Random Mode Loss Comparison',
        'demo/random_compare_losses.png',
        'Train updates',
    )

    save_plot(
        {
            'Basic(random)': h_r_basic['rewards'],
            'Double(random)': h_r_double['rewards'],
            'Dueling(random)': h_r_duel['rewards'],
            'Rainbow-lite(random)': h_r_rainbow['rewards'],
        },
        'Random Mode Reward Comparison',
        'demo/random_compare_rewards.png',
        'Episodes',
    )

    save_plot(
        {
            'Basic(player)': h_basic['rewards'],
            'Double(player)': h_double['rewards'],
            'Dueling(player)': h_duel['rewards'],
            'Rainbow(random)': h_rainbow['rewards'],
        },
        'Reward Comparison',
        'demo/reward_comparison.png',
        'Episodes',
    )

    with open('demo/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ep-hw1', dest='ep_hw1', type=int, default=1200)
    parser.add_argument('--ep-hw2', dest='ep_hw2', type=int, default=1800)
    parser.add_argument('--ep-hw3', dest='ep_hw3', type=int, default=2000)
    parser.add_argument('--ep-hw4', dest='ep_hw4', type=int, default=2200)
    parser.add_argument('--ep-cmp', dest='ep_cmp', type=int, default=1600)
    args = parser.parse_args()
    run_all(args)
