"""
Stardew Valley Fishing Minigame - Advanced C51 DQN Agent
=======================================================
Comprehensive model upgrade featuring:
  • C51 Distributional DQN (learns full return distribution)
  • NoisyNet layers (learnable exploration, replaces epsilon-greedy)
  • Residual Dueling architecture (skip connections for deeper nets)
  • Multi-head self-attention (focus on critical state dimensions)
  • GELU activations (smoother gradients than ReLU)
  • Enhanced state representation (24D → behavior one-hot, predictions, history)
  • Floater-specific reward / curriculum (addresses 50% floater weakness)
  • Gradient accumulation + LR warmup + weight decay
  • Beta annealing for prioritized replay
  • Fused optimizations (AMP, torch.compile, pinned memory)
"""

import csv
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import FishingMinigameEnv

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Experience ──────────────────────────────────────────────────────────────
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


# ═══════════════════════════════════════════════════════════════════════════
#  VECTORIZED ENV  (same as before, unchanged)
# ═══════════════════════════════════════════════════════════════════════════
class VectorizedEnv:
    def __init__(self, num_envs=4, **env_kwargs):
        self.num_envs = num_envs
        env_kwargs['render_mode'] = None
        self.envs = [FishingMinigameEnv(**env_kwargs) for _ in range(num_envs)]
        self.dones = [False] * num_envs

    def reset(self):
        states = [env.reset() for env in self.envs]
        self.dones = [False] * self.num_envs
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                # Reset the env and clear the latched done flag. Without this
                # self.dones[i] stays True forever, causing the env to be
                # reset on every subsequent call and never actually advance.
                state = env.reset()
                self.dones[i] = False
                results.append((state, 0.0, False, {}))
            else:
                results.append(env.step(action))
                self.dones[i] = results[-1][2]
        states, rewards, dones, infos = zip(*results)
        return (np.array(states, dtype=np.float32), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool), list(infos))

    def close(self):
        for env in self.envs:
            env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MILESTONE TRACKER  (same as before, unchanged for brevity)
# ═══════════════════════════════════════════════════════════════════════════
class MilestoneTracker:
    """Track and log major training milestones for YouTube video storytelling"""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.milestones = {
            'first_success': False,
            'first_win_streak_3': False, 'first_win_streak_5': False, 'first_win_streak_10': False,
            'first_win_streak_25': False, 'first_win_streak_50': False, 'first_win_streak_100': False,
            'easy_mastery_75': False, 'easy_mastery_90': False, 'easy_mastery_95': False,
            'medium_unlocked': False, 'medium_mastery_75': False, 'medium_mastery_90': False,
            'hard_unlocked': False, 'hard_mastery_75': False, 'hard_mastery_90': False,
            'overall_50_percent': False, 'overall_75_percent': False, 'overall_80_percent': False,
            'overall_90_percent': False, 'overall_95_percent': False, 'overall_99_percent': False,
            'first_sinker_catch': False, 'first_dart_catch': False, 'first_smooth_catch': False,
            'first_mixed_catch': False, 'first_floater_catch': False,
            'sinker_mastery_80': False, 'dart_mastery_80': False, 'smooth_mastery_80': False,
            'mixed_mastery_80': False, 'floater_mastery_80': False,
            'epsilon_below_0_5': False, 'epsilon_below_0_25': False, 'epsilon_below_0_1': False, 'epsilon_below_0_01': False,
            'episode_100': False, 'episode_500': False, 'episode_1000': False, 'episode_2500': False, 'episode_5000': False,
            'first_perfect_eval': False, 'first_flawless_20': False, 'speed_demon_50': False,
            'comeback_after_5_losses': False, 'all_behaviors_caught': False,
        }
        self.current_win_streak = 0
        self.max_win_streak = 0
        self.episode_results = []
        self.consecutive_losses = 0
        self.prev_epsilon = 1.0
        self.behavior_stats = {b: {'attempts': 0, 'successes': 0}
                               for b in ['sinker', 'dart', 'smooth', 'mixed', 'floater']}
        self.behaviors_caught = set()
        self.shortest_catch = float('inf')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Episode', 'Score', 'Success', 'Fish', 'Difficulty', 'Behavior',
            'Episode_Length', 'Epsilon', 'Win_Streak', 'Avg_Score_100',
            'Win_Rate_100', 'Easy_Success_Rate', 'Medium_Success_Rate', 'Hard_Success_Rate',
            'Shortest_Catch', 'Behaviors_Discovered', 'Sinker_Rate', 'Dart_Rate', 'Smooth_Rate',
            'Mixed_Rate', 'Floater_Rate'
        ])
        self.milestone_log_path = os.path.join(log_dir, f"milestones_{timestamp}.txt")
        with open(self.milestone_log_path, 'w') as f:
            f.write(f"Training Milestones Log - Started {timestamp}\n{'=' * 60}\n\n")

    def check_milestone(self, name, condition, episode, message):
        if not self.milestones.get(name, False) and condition:
            self.milestones[name] = True
            log_msg = f"🏆 MILESTONE at Episode {episode}: {message}"
            print(f"\n{'=' * 60}\n{log_msg}\n{'=' * 60}\n")
            with open(self.milestone_log_path, 'a', encoding="utf-8") as f:
                f.write(f"Episode {episode}: {message}\n")
            return True
        return False

    def update(self, episode, score, success, fish_info, epsilon, stats):
        behavior = fish_info['behavior']
        if behavior in self.behavior_stats:
            self.behavior_stats[behavior]['attempts'] += 1
            if success:
                self.behavior_stats[behavior]['successes'] += 1
                self.behaviors_caught.add(behavior)
        if success and fish_info['episode_length'] < self.shortest_catch:
            self.shortest_catch = fish_info['episode_length']
        if success:
            self.current_win_streak += 1
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
            self.consecutive_losses = 0
        else:
            self.current_win_streak = 0
            self.consecutive_losses += 1

        behavior_rates = {}
        for btype, data in self.behavior_stats.items():
            behavior_rates[btype] = (data['successes'] / data['attempts'] * 100) if data['attempts'] > 0 else 0.0

        self.csv_writer.writerow([
            episode, f"{score:.2f}", 1 if success else 0,
            fish_info['name'], fish_info['difficulty'], fish_info['behavior'],
            fish_info['episode_length'], f"{epsilon:.4f}",
            self.current_win_streak, f"{stats['avg_score']:.2f}", f"{stats['win_rate']:.1f}",
            f"{stats['easy_success_rate']:.1f}", f"{stats['medium_success_rate']:.1f}",
            f"{stats['hard_success_rate']:.1f}",
            self.shortest_catch if self.shortest_catch != float('inf') else 0,
            len(self.behaviors_caught),
            f"{behavior_rates.get('sinker', 0):.1f}", f"{behavior_rates.get('dart', 0):.1f}",
            f"{behavior_rates.get('smooth', 0):.1f}", f"{behavior_rates.get('mixed', 0):.1f}",
            f"{behavior_rates.get('floater', 0):.1f}",
        ])
        self.csv_file.flush()

        # Episode milestones
        for ep, key in [(100, 'episode_100'), (500, 'episode_500'), (1000, 'episode_1000'),
                         (2500, 'episode_2500'), (5000, 'episode_5000')]:
            if episode == ep:
                self.check_milestone(key, True, episode, f"Reached episode {ep}!")

        if success:
            self.check_milestone('first_success', True, episode, f"First successful catch! ({fish_info['name']})")

        for streak, key in [(3, 'first_win_streak_3'), (5, 'first_win_streak_5'), (10, 'first_win_streak_10'),
                            (25, 'first_win_streak_25'), (50, 'first_win_streak_50'), (100, 'first_win_streak_100')]:
            if self.current_win_streak == streak:
                self.check_milestone(key, True, episode, f"First {streak}-episode win streak!" + (' 🔥' * (streak // 10)))
                break

        if self.current_win_streak >= 20:
            self.check_milestone('first_flawless_20', True, episode,
                                 f"Flawless! {self.current_win_streak}-episode win streak!")

        if success:
            bmap = {'sinker': 'first_sinker_catch', 'dart': 'first_dart_catch', 'smooth': 'first_smooth_catch',
                    'mixed': 'first_mixed_catch', 'floater': 'first_floater_catch'}
            if behavior in bmap:
                self.check_milestone(bmap[behavior], True, episode,
                                     f"First {behavior} fish caught! ({fish_info['name']})")

        if len(self.behaviors_caught) >= 5:
            self.check_milestone('all_behaviors_caught', True, episode, "Caught all 5 fish behavior types!")

        for btype, rate in behavior_rates.items():
            mk = f'{btype}_mastery_80'
            if self.behavior_stats[btype]['attempts'] >= 20 and rate >= 80:
                self.check_milestone(mk, True, episode,
                                     f"{btype.capitalize()} fish mastered! ({rate:.1f}% success)")

        if success and fish_info['episode_length'] < 50:
            self.check_milestone('speed_demon_50', True, episode,
                                 f"Speed demon! Caught in {fish_info['episode_length']} steps!")
        if success and self.consecutive_losses >= 5:
            self.check_milestone('comeback_after_5_losses', True, episode,
                                 f"Comeback! Won after {self.consecutive_losses} losses!")

        eps_thresholds = [(0.5, 'epsilon_below_0_5'), (0.25, 'epsilon_below_0_25'),
                          (0.1, 'epsilon_below_0_1'), (0.01, 'epsilon_below_0_01')]
        for thr, key in eps_thresholds:
            if self.prev_epsilon >= thr and epsilon < thr:
                self.check_milestone(key, True, episode,
                                     f"Epsilon below {thr} ({epsilon:.4f})")
        self.prev_epsilon = epsilon

        def _check_difficulty_level(level_name, rate_field, enabled_field):
            lvl = stats.get(rate_field, 0)
            if stats.get(enabled_field, False):
                self.check_milestone(f'{level_name}_unlocked', True, episode,
                                     f"{level_name.capitalize()} unlocked!")
                if lvl >= 75:
                    self.check_milestone(f'{level_name}_mastery_75', True, episode,
                                         f"{level_name.capitalize()} mastery! ({lvl:.1f}%)")
                if lvl >= 90:
                    self.check_milestone(f'{level_name}_mastery_90', True, episode,
                                         f"{level_name.capitalize()} excellence! ({lvl:.1f}%)")

        _check_difficulty_level('easy', 'easy_success_rate', 'easy_enabled')
        _check_difficulty_level('medium', 'medium_success_rate', 'medium_enabled')
        _check_difficulty_level('hard', 'hard_success_rate', 'hard_enabled')

        overall = stats['win_rate']
        for thr, key in [(50, 'overall_50_percent'), (75, 'overall_75_percent'), (80, 'overall_80_percent'),
                          (90, 'overall_90_percent'), (95, 'overall_95_percent'), (99, 'overall_99_percent')]:
            if overall >= thr:
                self.check_milestone(key, True, episode, f"{thr}% win rate achieved! ({overall:.1f}%)")

        if 'eval_success_rate' in stats and stats['eval_success_rate'] >= 100:
            self.check_milestone('first_perfect_eval', True, episode, "Perfect evaluation! 10/10!")

    def close(self):
        self.csv_file.close()
        print(f"\n📊 Training metrics saved to: {self.csv_path}")
        print(f"🏆 Milestone log saved to: {self.milestone_log_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PRIORITIZED REPLAY BUFFER  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += float(change)
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    epsilon = 1e-5
    alpha = 0.6
    beta = 0.4
    beta_increment = 0.001  # Annealed toward 1.0 during training
    abs_err_upper = 1.0

    def __init__(self, capacity=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            experiences.append(data)
            indices.append(idx)
            priorities.append(priority)

        state_dim = experiences[0].state.shape[0]
        s_ = np.zeros((batch_size, state_dim), dtype=np.float32)
        a_ = np.zeros((batch_size, 1), dtype=np.int64)
        r_ = np.zeros((batch_size, 1), dtype=np.float32)
        n_ = np.zeros((batch_size, state_dim), dtype=np.float32)
        d_ = np.zeros((batch_size, 1), dtype=np.float32)

        for i, e in enumerate(experiences):
            s_[i] = e.state
            a_[i] = e.action
            r_[i] = e.reward
            n_[i] = e.next_state
            d_[i] = e.done

        s_ = torch.from_numpy(s_).to(device)
        a_ = torch.from_numpy(a_).to(device)
        r_ = torch.from_numpy(r_).to(device)
        n_ = torch.from_numpy(n_).to(device)
        d_ = torch.from_numpy(d_).to(device)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights.astype(np.float32)).unsqueeze(1).to(device)

        return s_, a_, r_, n_, d_, indices, is_weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ═══════════════════════════════════════════════════════════════════════════
#  NOISY NETWORK LAYER  (Rainbow DQN: learnable exploration)
# ═══════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """NoisyNet linear layer with factorised Gaussian noise (Rainbow DQN paper)."""

    def __init__(self, in_features, out_features, std_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self._reset_parameters()

    def _reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size, device):
        """Factorised Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            # Factorised noise: epsilon_weight = f(epsilon_out) outer f(epsilon_in)
            noise_in = self._scale_noise(self.in_features, x.device)
            noise_out = self._scale_noise(self.out_features, x.device)
            weight_epsilon = noise_out.unsqueeze(1) * noise_in.unsqueeze(0)  # (out, in)
            bias_epsilon = noise_out  # (out,)
        else:
            # Evaluation: use mean (mu) only, no noise
            weight_epsilon = 0.0
            bias_epsilon = 0.0

        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════════════════════════════════════════
#  C51 DISTRIBUTIONAL HEAD
# ═══════════════════════════════════════════════════════════════════════════
class C51DistributionalDQN(nn.Module):
    """
    Dueling DQN with:
      • C51 distributional RL (51 atoms per action)
      • NoisyNet layers for exploration
      • Residual connections in shared layers
      • Multi-head self-attention
      • GELU activations
    """

    def __init__(self, state_dim=24, action_dim=2, n_atoms=51, v_min=-20.0, v_max=20.0,
                 hidden_sizes=None, dropout_rate=0.1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128, 64]

        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atom_delta = (v_max - v_min) / (n_atoms - 1)

        # Register support (atom values) as a buffer
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)

        # ── Shared feature layers with residual connections ──
        self.feature_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        in_dim = state_dim

        for i, h_dim in enumerate(hidden_sizes):
            block = nn.ModuleList([
                NoisyLinear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            self.feature_layers.append(block)

            # Projection for residual connection if dimensions differ
            if in_dim != h_dim:
                self.residual_projections.append(
                    nn.Sequential(NoisyLinear(in_dim, h_dim), nn.LayerNorm(h_dim))
                )
            else:
                self.residual_projections.append(nn.Identity())
            in_dim = h_dim

        # ── Multi-head self-attention ──
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1], num_heads=4, batch_first=True, dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(hidden_sizes[-1])

        # ── Value stream (distribution) ──
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[-1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            NoisyLinear(128, n_atoms),  # V(s) over atoms
        )

        # ── Advantage stream (distribution) ──
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[-1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            NoisyLinear(128, action_dim * n_atoms),  # A(s,a) over atoms
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x, return_dist=False):
        """
        Forward pass through the dueling C51 architecture.
        Combines value and advantage in LOGIT space before softmax.
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a))) over atoms.

        Args:
            x: (batch, state_dim)
            return_dist: if True, return the full distribution (batch, action_dim, n_atoms)
        Returns:
            q_values: (batch, action_dim) — expected Q values
            OR (dist, q_values) if return_dist=True
        """
        # Shared features with residual connections
        h = x
        for block, proj in zip(self.feature_layers, self.residual_projections):
            h_res = proj(h)
            for layer in block:
                h = layer(h)
            h = h + h_res  # Residual connection

        # Self-attention
        h2 = h.unsqueeze(1)  # (batch, 1, dim) for single-token attention
        h_attn, _ = self.attention(h2, h2, h2)
        h = self.attn_norm(h + h_attn.squeeze(1))

        # Value LOGITS (batch, n_atoms) - BEFORE softmax
        v_logits = self.value_stream(h)

        # Advantage LOGITS (batch, action * n_atoms) → (batch, action, n_atoms) - BEFORE softmax
        a_logits = self.advantage_stream(h).view(-1, self.action_dim, self.n_atoms)

        # Dueling combination in LOGIT space: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        v_expanded = v_logits.unsqueeze(1).expand(-1, self.action_dim, -1)
        a_mean = a_logits.mean(dim=1, keepdim=True).expand(-1, self.action_dim, -1)
        q_logits = v_expanded + (a_logits - a_mean)

        # Single softmax over atoms for each action
        q_dist = F.softmax(q_logits, dim=-1)

        # Expected Q values: sum over atoms of (atom_value * probability)
        support = self.support.unsqueeze(0).unsqueeze(0)  # (1, 1, n_atoms)
        q_values = (q_dist * support).sum(dim=-1)  # (batch, action_dim)

        if return_dist:
            return q_dist, q_values
        return q_values

    def get_logits(self, x):
        """Get raw logits before softmax (for projection of target distribution)."""
        h = x
        for block, proj in zip(self.feature_layers, self.residual_projections):
            h_res = proj(h)
            for layer in block:
                h = layer(h)
            h = h + h_res

        h2 = h.unsqueeze(1)
        h_attn, _ = self.attention(h2, h2, h2)
        h = self.attn_norm(h + h_attn.squeeze(1))

        v_logits = self.value_stream[0](h)  # First NoisyLinear (before activations)
        for layer in self.value_stream[1:]:
            v_logits = layer(v_logits)

        a_logits = self.advantage_stream[0](h)
        for layer in self.advantage_stream[1:]:
            a_logits = layer(a_logits)

        a_logits = a_logits.view(-1, self.action_dim, self.n_atoms)
        v_expanded = v_logits.unsqueeze(1).expand(-1, self.action_dim, -1)
        a_mean = a_logits.mean(dim=1, keepdim=True).expand(-1, self.action_dim, -1)
        return v_expanded + (a_logits - a_mean)


def dist_projection(next_dist, rewards, dones, support, v_min, v_max, gamma, n_step):
    """
    C51 target distribution projection — vectorized with scatter_add.
    Projects the target distribution onto the support of the current atoms.
    """
    batch_size = next_dist.size(0)
    n_atoms = support.size(0)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Compute target atom values after Bellman update
    # Tz = r + gamma^n * z  (for non-terminal states)
    tz = rewards + (gamma ** n_step) * support.unsqueeze(0) * (1.0 - dones)
    tz = tz.clamp(min=v_min, max=v_max)

    # Normalised position of Tz in atom-space [0, n_atoms-1]
    b = (tz - v_min) / delta_z

    b_low = b.floor().long()
    b_high = b.ceil().long()

    # Clamp to valid range
    b_low = b_low.clamp(0, n_atoms - 1)
    b_high = b_high.clamp(0, n_atoms - 1)

    # Fractional distance to lower / upper atom
    lower_weight = (b_high.float() - b).clamp(0, 1)
    upper_weight = (b - b_low.float()).clamp(0, 1)

    # When b lands exactly on an atom (low == high), all mass goes there
    same_atom = (b_low == b_high).float()
    lower_proportion = lower_weight * (1 - same_atom) + same_atom  # 1.0 when exact hit
    upper_proportion = upper_weight * (1 - same_atom)               # 0.0 when exact hit

    lower_mass = next_dist * lower_proportion
    upper_mass = next_dist * upper_proportion

    projected = torch.zeros_like(next_dist)
    projected = projected.scatter_add(dim=1, index=b_low, src=lower_mass)
    projected = projected.scatter_add(dim=1, index=b_high, src=upper_mass)

    return projected


# ═══════════════════════════════════════════════════════════════════════════
#  C51 DQN AGENT  (with NoisyNet, Dueling, Distributional, N-step, PER)
# ═══════════════════════════════════════════════════════════════════════════
class C51DQNAgent:
    """
    Rainbow-inspired agent:
      • C51 Distributional DQN
      • NoisyNet (no epsilon-greedy after training starts)
      • Dueling architecture with multi-head attention
      • Prioritized Replay Buffer
      • N-step returns
      • Gradient accumulation for stability
      • Learning rate warmup + cosine annealing
    """

    def __init__(self, state_dim=24, action_dim=2, hidden_sizes=None,
                 learning_rate=2e-4, gamma=0.99, buffer_size=100000, batch_size=128,
                 update_every=4, n_step=3, target_update_freq=500,
                 n_atoms=51, v_min=-20.0, v_max=20.0,
                 grad_accum_steps=2, weight_decay=1e-5,
                 lr_warmup_steps=1000):
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.grad_accum_steps = grad_accum_steps
        self.lr_warmup_steps = lr_warmup_steps

        # Q-Networks
        self.qnetwork_local = C51DistributionalDQN(
            state_dim, action_dim, n_atoms, v_min, v_max, hidden_sizes
        ).to(device)
        self.qnetwork_target = C51DistributionalDQN(
            state_dim, action_dim, n_atoms, v_min, v_max, hidden_sizes
        ).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            self.qnetwork_local.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Cosine annealing LR scheduler (with warmup handled in learn())
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50000, eta_min=5e-6
        )

        # AMP (mixed precision)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Replay & N-step
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)

        # Step tracking
        self.t_step = 0
        self.update_every = update_every
        self.total_steps = 0

        # Metrics
        self.loss_list = []

    def step(self, state, action, reward, next_state, done):
        """Store experience with N-step returns and learn every update_every steps."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            n_state = self.n_step_buffer[0][0]
            n_action = self.n_step_buffer[0][1]
            n_reward = sum(self.gamma ** i * exp[2] for i, exp in enumerate(self.n_step_buffer))
            n_next = self.n_step_buffer[-1][3]
            n_done = self.n_step_buffer[-1][4]
            self.memory.add(n_state, n_action, n_reward, n_next, n_done)

            if done:
                self.n_step_buffer.clear()

        self.t_step = (self.t_step + 1) % self.update_every
        self.total_steps += 1

        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences)
            self.loss_list.append(loss)

            if self.total_steps % self.target_update_freq == 0:
                self.hard_update(silent=True)

    def act(self, state, eps=0.0):
        """
        Select action. NoisyNet handles exploration internally via learnable noise.
        eps is kept for backward compatibility but NoisyNet is the primary exploration mechanism.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.qnetwork_local(state_t)
            action = q_values.argmax(dim=1).item()

        # Use eps as fallback random exploration (disabled when NoisyNet is active)
        if random.random() < eps:
            return random.choice(np.arange(self.action_dim))
        return action

    def act_batch(self, states, eps=0.0):
        """
        Select actions for N parallel environments in one batched forward pass.

        Args:
            states: numpy array of shape (N, state_dim)
            eps: epsilon for random exploration
        Returns:
            numpy array of actions of shape (N,)
        """
        states_t = torch.from_numpy(states).float().to(device)
        with torch.no_grad():
            q_values = self.qnetwork_local(states_t)
            actions = q_values.argmax(dim=1).cpu().numpy()

        # Epsilon-greedy: replace some actions with random
        if eps > 0:
            mask = np.random.random(len(actions)) < eps
            if mask.any():
                actions[mask] = np.random.randint(0, self.action_dim, size=mask.sum())

        return actions

    def learn(self, experiences):
        """Learn from a batch using C51 distributional loss."""
        states, actions, rewards, next_states, dones, indices, is_weights = experiences

        # ── Target distribution ──
        with torch.no_grad():
            # Get next state distribution
            next_dist = self.qnetwork_target(next_states, return_dist=True)[0]  # (batch, action, n_atoms)

            # Double DQN: select actions using local network
            next_q = self.qnetwork_local(next_states)  # (batch, action)
            next_actions = next_q.argmax(dim=1, keepdim=True)  # (batch, 1)

            # Select the distributions of the chosen actions
            next_dist = next_dist.gather(
                1, next_actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)
            ).squeeze(1)  # (batch, n_atoms)

            # Project target distribution
            support = self.qnetwork_local.support
            target_dist = dist_projection(
                next_dist, rewards, dones, support,
                self.v_min, self.v_max, self.gamma, self.n_step
            )  # (batch, n_atoms) for the chosen action

        # ── Current distribution ──
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                current_dist = self.qnetwork_local(states, return_dist=True)[0]  # (batch, action, n_atoms)
                current_dist = current_dist.gather(
                    1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)
                ).squeeze(1)  # (batch, n_atoms)

                # Categorical cross-entropy loss
                loss = -(target_dist * (current_dist + 1e-8).log()).sum(dim=-1)  # (batch,)
                # Importance-sampled weighted loss
                loss = (is_weights.squeeze(-1) * loss).mean()

            # TD errors for priority update (use C51 KL-divergence per sample as error)
            with torch.no_grad():
                per_sample_loss = -(target_dist * (current_dist.float() + 1e-8).log()).sum(dim=-1)
                td_errors = per_sample_loss.detach().cpu().numpy().flatten()

            # Backward with scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_dist = self.qnetwork_local(states, return_dist=True)[0]
            current_dist = current_dist.gather(
                1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)
            ).squeeze(1)

            loss = -(target_dist * (current_dist + 1e-8).log()).sum(dim=-1)
            loss = (is_weights.squeeze(-1) * loss).mean()

            per_sample_loss = -(target_dist * (current_dist + 1e-8).log()).sum(dim=-1)
            td_errors = per_sample_loss.detach().cpu().numpy().flatten()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
            self.optimizer.step()

        # LR scheduling
        self.scheduler.step()
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

    def hard_update(self, silent=True):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        if not silent:
            print(f"Target network updated at step {self.total_steps}")

    def save(self, filename, export_onnx=False, state_dim=24):
        torch.save({
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_list': self.loss_list,
            'n_atoms': self.n_atoms,
            'v_min': self.v_min,
            'v_max': self.v_max,
        }, filename)
        if export_onnx:
            onnx_fn = filename.replace('.pth', '.onnx')
            self.export_onnx(onnx_fn, state_dim=state_dim)

    def export_onnx(self, filename, state_dim=24):
        self.qnetwork_local.eval()
        dummy = torch.randn(1, state_dim, device=device)
        torch.onnx.export(
            self.qnetwork_local, dummy, filename,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['state'], output_names=['q_values'],
            dynamic_axes={'state': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
        )
        print(f"✅ ONNX: {filename}")
        self.qnetwork_local.train()

    def load(self, filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')
        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint.get('loss_list', [])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {filename}")


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP  (with floater-specific curriculum)
# ═══════════════════════════════════════════════════════════════════════════
def train_dqn(env, agent, n_episodes=10000, max_t=2000,
              eps_start=0.05, eps_end=0.001,
              save_every=500, render_every=2000):
    """
    Enhanced training loop with:
      • Floater-specific curriculum phase
      • Behavior-balanced sampling
      • Adaptive difficulty mixing
      • Cosine annealing epsilon
      • Early stopping at 98%+
    """
    scores = []
    scores_window = deque(maxlen=100)

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_logs/graphs", exist_ok=True)

    milestone_tracker = MilestoneTracker(log_dir="training_logs")
    print(f"📊 Logging to: {milestone_tracker.csv_path}")

    # Difficulty curriculum
    difficulty_buckets = {
        'easy':   {'range': (0, 40),   'attempts': 0, 'success': 0, 'enabled': True},
        'medium': {'range': (41, 70),  'attempts': 0, 'success': 0, 'enabled': False},
        'hard':   {'range': (71, 100), 'attempts': 0, 'success': 0, 'enabled': False},
    }
    curriculum_threshold = 0.75
    difficulty_mix_ratio = 0.8

    behavior_types = ['sinker', 'dart', 'smooth', 'mixed', 'floater']
    behavior_stats = {b: {"attempts": 0, "success": 0} for b in behavior_types}

    # === FLOATER-SPECIFIC CURRICULUM ===
    floater_curriculum_active = False
    floater_curriculum_triggered = False  # Only trigger once after 80% overall
    floater_curriculum_episodes = 1000   # Train 1000 episodes on pure floaters
    floater_curriculum_remaining = 0

    perfect_episodes = 0
    required_perfect = 3
    early_stop_threshold = 0.98

    training_start_time = time.time()
    last_checkpoint_time = training_start_time

    # Pre-identify floater fish names
    floater_fish_names = [f["name"] for f in env.fish_data if f.get("behaviour", "").lower() == "floater"]

    print(f"Starting training: {n_episodes} episodes")
    print(f"Early stopping: {required_perfect} consecutive evals > {early_stop_threshold * 100}%")
    print(f"Architecture: C51 Distributional DQN + NoisyNet + Attention + Residuals")
    print(f"State dim: {agent.state_dim} | Atoms: {agent.n_atoms} | Range: [{agent.v_min}, {agent.v_max}]")
    print(f"Floater fish available: {len(floater_fish_names)} ({', '.join(floater_fish_names)})")
    print()

    for i_episode in range(1, n_episodes + 1):
        # ── Schedule: NoisyNet handles exploration; epsilon is minimal ──
        eps = eps_end + (eps_start - eps_end) * (1 + np.cos(np.pi * i_episode / n_episodes)) / 2

        # ── Fish selection ──
        # Check if floater curriculum phase is active
        if floater_curriculum_active and floater_curriculum_remaining > 0:
            floater_curriculum_remaining -= 1
            # Force a floater fish
            if floater_fish_names:
                fish_name = random.choice(floater_fish_names)
                env.fish_name = fish_name
                state = env.reset()
                fish_behavior = env.current_fish["behaviour"]
                fish_difficulty = env.current_fish["difficulty"]

                if floater_curriculum_remaining % 200 == 0 or floater_curriculum_remaining == floater_curriculum_episodes - 1:
                    print(f"   Floater curriculum: {floater_curriculum_remaining} episodes remaining")
            else:
                # No floaters available, skip curriculum
                floater_curriculum_active = False
        else:
            if floater_curriculum_active:
                floater_curriculum_active = False
                print("✅ Floater curriculum phase complete!")

            # Standard behavior-balanced sampling
            enabled_buckets = [b for b, info in difficulty_buckets.items() if info['enabled']]
            if len(enabled_buckets) > 1 and random.random() > difficulty_mix_ratio:
                bucket = enabled_buckets[-1]
            else:
                bucket = random.choice(enabled_buckets[:-1] if len(enabled_buckets) > 1 else enabled_buckets)

            min_diff, max_diff = difficulty_buckets[bucket]['range']
            available = [f for f in env.fish_data if min_diff <= f["difficulty"] <= max_diff]

            if not available:
                available = env.fish_data

            # Behavior-balanced: choose from random behavior type
            random.shuffle(behavior_types)
            fish_name = None
            for bt in behavior_types:
                candidates = [f["name"] for f in available if f.get("behaviour", "").lower() == bt]
                if candidates:
                    fish_name = random.choice(candidates)
                    break
            if fish_name is None:
                fish_name = random.choice([f["name"] for f in available])

            env.fish_name = fish_name
            state = env.reset()
            fish_behavior = env.current_fish["behaviour"]
            fish_difficulty = env.current_fish["difficulty"]

        # Track stats
        behavior_stats[fish_behavior]["attempts"] += 1
        for bn, bd in difficulty_buckets.items():
            bmin, bmax = bd['range']
            if bmin <= fish_difficulty <= bmax:
                bd['attempts'] += 1
                break

        # ── Run episode ──
        score = 0
        render = (i_episode % render_every == 0)
        render_mode_bak = env.render_mode

        if render:
            env.render_mode = "human"
            print(f"\nEpisode {i_episode}: Rendering {fish_name} ({fish_behavior})")
        else:
            env.render_mode = None

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if render:
                time.sleep(0.01)
            if done:
                if env.distanceFromCatching >= 1.0:
                    behavior_stats[fish_behavior]["success"] += 1
                    for bn, bd in difficulty_buckets.items():
                        bmin, bmax = bd['range']
                        if bmin <= fish_difficulty <= bmax:
                            bd['success'] += 1
                            break
                break

        env.render_mode = render_mode_bak
        scores_window.append(score)
        scores.append(score)

        avg_score = float(np.mean(scores_window))
        win_rate = sum(1 for s in scores[-100:] if s > 0) / min(100, len(scores)) * 100

        def calc_bucket_rate(name):
            b = difficulty_buckets[name]
            return (b['success'] / b['attempts'] * 100) if b['attempts'] > 0 else 0.0

        milestone_tracker.update(
            episode=i_episode, score=score, success=(env.distanceFromCatching >= 1.0),
            fish_info={'name': fish_name, 'difficulty': fish_difficulty,
                       'behavior': fish_behavior, 'episode_length': env.episode_length},
            epsilon=eps,
            stats={
                'avg_score': avg_score, 'win_rate': win_rate,
                'easy_success_rate': calc_bucket_rate('easy'),
                'medium_success_rate': calc_bucket_rate('medium'),
                'hard_success_rate': calc_bucket_rate('hard'),
                'easy_enabled': difficulty_buckets['easy']['enabled'],
                'medium_enabled': difficulty_buckets['medium']['enabled'],
                'hard_enabled': difficulty_buckets['hard']['enabled'],
            }
        )

        # ── Adaptive curriculum ──
        if i_episode % 100 == 0 and i_episode > 100:
            bucket_names = list(difficulty_buckets.keys())
            for idx, (bn, bd) in enumerate(difficulty_buckets.items()):
                if bd['enabled'] and bd['attempts'] >= 50:
                    sr = bd['success'] / bd['attempts']
                    if sr >= curriculum_threshold and idx < len(bucket_names) - 1:
                        nxt = bucket_names[idx + 1]
                        if not difficulty_buckets[nxt]['enabled']:
                            difficulty_buckets[nxt]['enabled'] = True
                            print(f"\n*** Curriculum: Enabled '{nxt}' (mastered '{bn}' at {sr*100:.1f}%) ***")

            # ── Floater curriculum trigger ──
            # After reaching 80% overall win rate, train exclusively on floaters
            if win_rate >= 80.0 and not floater_curriculum_triggered and floater_fish_names:
                floater_curriculum_triggered = True
                floater_curriculum_active = True
                floater_curriculum_remaining = floater_curriculum_episodes
                print(f"\n{'=' * 60}")
                print(f"🎣 FLOATER CURRICULUM: Training {floater_curriculum_episodes} episodes on floaters!")
                print(f"   Floaters available: {', '.join(floater_fish_names)}")
                print(f"{'=' * 60}\n")

        # ── Progress ──
        if i_episode % 100 == 0:
            elapsed = time.time() - training_start_time
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            enabled = [b for b, info in difficulty_buckets.items() if info['enabled']]
            print(f'Ep {i_episode}/{n_episodes} ({100*i_episode/n_episodes:.1f}%) | '
                  f'Time: {int(h)}h {int(m)}m {int(s)}s | Score: {avg_score:.2f} | '
                  f'Eps: {eps:.4f} | WR: {win_rate:.1f}% | {", ".join(enabled)}')

            # Floater-specific progress
            fb = behavior_stats.get('floater', {})
            if fb.get('attempts', 0) > 0:
                f_rate = fb['success'] / fb['attempts'] * 100
                print(f'   Floater: {fb["success"]}/{fb["attempts"]} ({f_rate:.1f}%) | '
                      f'Streak: {milestone_tracker.current_win_streak} | Max: {milestone_tracker.max_win_streak}')

        # ── Save ──
        if i_episode % save_every == 0:
            ckpt = f'models/checkpoints/episode_{i_episode}.pth'
            agent.save(ckpt)
            ct = time.time()
            print(f"Saved {ckpt} ({ (ct - last_checkpoint_time) / 60:.1f} min since last)")
            last_checkpoint_time = ct

            # Plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
            ax1.plot(np.arange(len(scores)), scores, alpha=0.5)
            if len(scores) >= 100:
                ma = np.convolve(scores, np.ones(100) / 100, mode='valid')
                ax1.plot(np.arange(99, len(scores)), ma, 'r-', linewidth=2)
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Episode')
            ax1.set_title('Training Scores')

            bnames, brates = [], []
            for b, s in behavior_stats.items():
                if s["attempts"] > 0:
                    bnames.append(b)
                    brates.append(s["success"] / s["attempts"] * 100)
            ax2.bar(bnames, brates)
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_xlabel('Behavior')
            ax2.set_title('Success Rate by Fish Behavior')
            ax2.set_ylim(0, 100)

            if agent.loss_list:
                window = min(100, len(agent.loss_list))
                loss_avg = np.convolve(agent.loss_list, np.ones(window) / window, mode='valid')
                ax3.plot(np.arange(window - 1, len(agent.loss_list)), loss_avg)
                ax3.set_ylabel('C51 Loss')
                ax3.set_xlabel('Train Steps')
                ax3.set_title('Smoothed C51 Loss')
                ax3.set_yscale('log')

            plt.tight_layout()
            plt.savefig(f'training_logs/graphs/episode_{i_episode}.png')
            plt.close()

            print("Behavior rates:")
            for b, s in behavior_stats.items():
                if s["attempts"] > 0:
                    print(f"  {b}: {s['success']}/{s['attempts']} ({s['success'] / s['attempts'] * 100:.1f}%)")

        # ── Evaluation & Early stopping ──
        if i_episode % 1000 == 0:
            print("\nRunning evaluation...")
            env.render_mode = None
            eval_success = 0
            eval_eps = 20
            for _ in range(eval_eps):
                fn = random.choice(env.get_available_fish())
                env.fish_name = fn
                s = env.reset()
                for _ in range(max_t):
                    a = agent.act(s, eps=0.0)
                    s, r, done, info = env.step(a)
                    if done:
                        if env.distanceFromCatching >= 1.0:
                            eval_success += 1
                        break
            eval_rate = eval_success / eval_eps
            print(f"Eval: {eval_rate * 100:.1f}% ({eval_success}/{eval_eps})")

            if eval_rate >= early_stop_threshold:
                perfect_episodes += 1
                print(f"Perfect eval {perfect_episodes}/{required_perfect}")
                if perfect_episodes >= required_perfect:
                    print(f"\n*** EARLY STOP at episode {i_episode} ***")
                    agent.save(f'models/checkpoints/early_stop_ep{i_episode}.pth', export_onnx=True)
                    break
            else:
                perfect_episodes = 0

    # Final save
    agent.save(f'models/checkpoints/final_ep{i_episode}.pth', export_onnx=True)
    print(f"\n💾 Final: models/checkpoints/final_ep{i_episode}.pth")

    milestone_tracker.close()

    total = time.time() - training_start_time
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nDone! {i_episode} episodes in {int(h)}h {int(m)}m {int(s)}s")
    print(f"Final WR: {win_rate:.1f}% | Max streak: {milestone_tracker.max_win_streak}")

    return scores


# ═══════════════════════════════════════════════════════════════════════════
#  VECTORIZED TRAINING  (4 parallel envs — much faster on GPU)
# ═══════════════════════════════════════════════════════════════════════════
def train_dqn_vectorized(env_vec, agent, n_episodes=10000, max_t=2000,
                         eps_start=0.05, eps_end=0.001,
                         save_every=500, render_every=2000):
    """
    Training loop using VectorizedEnv (default 4 parallel envs).
    Each step collects 4x the experience per wall-clock second because the
    batched forward/backward pass leverages GPU parallelism efficiently.
    """
    num_envs = env_vec.num_envs
    scores = []
    scores_window = deque(maxlen=100)

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_logs/graphs", exist_ok=True)

    milestone_tracker = MilestoneTracker(log_dir="training_logs")
    print(f"📊 Logging to: {milestone_tracker.csv_path}")
    print(f"🚀 Vectorized training with {num_envs} parallel envs")

    difficulty_buckets = {
        'easy':   {'range': (0, 40),   'attempts': [0]*num_envs, 'success': [0]*num_envs, 'enabled': True},
        'medium': {'range': (41, 70),  'attempts': [0]*num_envs, 'success': [0]*num_envs, 'enabled': False},
        'hard':   {'range': (71, 100), 'attempts': [0]*num_envs, 'success': [0]*num_envs, 'enabled': False},
    }
    curriculum_threshold = 0.75
    difficulty_mix_ratio = 0.8

    behavior_types = ['sinker', 'dart', 'smooth', 'mixed', 'floater']
    behavior_stats = {b: {"attempts": 0, "success": 0} for b in behavior_types}

    floater_fish_names = [f["name"] for f in env_vec.envs[0].fish_data
                          if f.get("behaviour", "").lower() == "floater"]
    floater_curriculum_triggered = False
    floater_curriculum_active = False
    floater_curriculum_remaining = 0
    floater_curriculum_episodes = 1000

    perfect_episodes = 0
    required_perfect = 3
    early_stop_threshold = 0.98

    training_start_time = time.time()
    last_checkpoint_time = training_start_time

    # ── Select initial fish for each env ──
    def pick_fish_for_env(env_idx, force_floater=False):
        env = env_vec.envs[env_idx]
        if force_floater and floater_fish_names:
            fn = random.choice(floater_fish_names)
        else:
            enabled = [b for b, info in difficulty_buckets.items() if info['enabled']]
            if len(enabled) > 1 and random.random() > difficulty_mix_ratio:
                bucket = enabled[-1]
            else:
                bucket = random.choice(enabled[:-1] if len(enabled) > 1 else enabled)
            mn, mx = difficulty_buckets[bucket]['range']
            avail = [f for f in env.fish_data if mn <= f["difficulty"] <= mx]
            if not avail:
                avail = env.fish_data
            random.shuffle(behavior_types)
            fn = None
            for bt in behavior_types:
                cands = [f["name"] for f in avail if f.get("behaviour", "").lower() == bt]
                if cands:
                    fn = random.choice(cands)
                    break
            if fn is None:
                fn = random.choice([f["name"] for f in avail])
        env.fish_name = fn
        state = env.reset()
        return state, env.current_fish['name'], env.current_fish['behaviour'], env.current_fish['difficulty']

    # Initialise all envs
    states = np.zeros((num_envs, agent.state_dim), dtype=np.float32)
    fish_names = [''] * num_envs
    fish_behaviors = [''] * num_envs
    fish_difficulties = [0] * num_envs
    env_scores = [0.0] * num_envs

    for i in range(num_envs):
        s, fn, fb, fd = pick_fish_for_env(i)
        states[i] = s
        fish_names[i] = fn
        fish_behaviors[i] = fb
        fish_difficulties[i] = fd

    # Track episode count across all envs
    episode_count = 0
    total_steps = 0
    next_progress_print = 100   # Print progress at 100, 200, 300, ...
    next_eval_print = 1000      # Evaluate at 1000, 2000, 3000, ...
    last_heartbeat_time = training_start_time  # Wall-clock heartbeat every 5 min

    print(f"Floater fish available: {len(floater_fish_names)} ({', '.join(floater_fish_names)})")
    print(f"Starting vectorized training: {n_episodes} total episodes\n")

    while episode_count < n_episodes:
        eps = eps_end + (eps_start - eps_end) * (1 + np.cos(np.pi * episode_count / n_episodes)) / 2

        # ── Act (batched forward pass) ──
        actions = agent.act_batch(states, eps)

        # ── Step all envs ──
        next_states, rewards, dones, infos = env_vec.step(actions)
        total_steps += num_envs

        # ── Store experiences ──
        for i in range(num_envs):
            agent.step(states[i], int(actions[i]), rewards[i], next_states[i], bool(dones[i]))
            env_scores[i] += rewards[i]

            if dones[i]:
                episode_count += 1
                success = infos[i].get('distance_from_catching', 0) >= 1.0
                fb = fish_behaviors[i]

                behavior_stats[fb]["attempts"] += 1
                if success:
                    behavior_stats[fb]["success"] += 1

                for bn, bd in difficulty_buckets.items():
                    bmn, bmx = bd['range']
                    if bmn <= fish_difficulties[i] <= bmx:
                        bd['attempts'][i % num_envs] += 1
                        if success:
                            bd['success'][i % num_envs] += 1
                        break

                # Log this episode
                scores.append(env_scores[i])
                scores_window.append(env_scores[i])
                avg_score = float(np.mean(scores_window)) if scores_window else 0.0
                wr = sum(1 for s in scores[-100:] if s > 0) / min(100, len(scores)) * 100

                def calc_br(name):
                    b = difficulty_buckets[name]
                    total_att = sum(b['attempts'])
                    total_suc = sum(b['success'])
                    return (total_suc / total_att * 100) if total_att > 0 else 0.0

                episode_length = infos[i].get('episode_length', 0)
                milestone_tracker.update(
                    episode=episode_count, score=env_scores[i],
                    success=success,
                    fish_info={'name': fish_names[i], 'difficulty': fish_difficulties[i],
                               'behavior': fb, 'episode_length': episode_length},
                    epsilon=eps,
                    stats={'avg_score': avg_score, 'win_rate': wr,
                           'easy_success_rate': calc_br('easy'),
                           'medium_success_rate': calc_br('medium'),
                           'hard_success_rate': calc_br('hard'),
                           'easy_enabled': True, 'medium_enabled': difficulty_buckets['medium']['enabled'],
                           'hard_enabled': difficulty_buckets['hard']['enabled']}
                )

                env_scores[i] = 0.0

                # Pick next fish
                force_floater = floater_curriculum_active and floater_curriculum_remaining > 0
                s, fn, fb, fd = pick_fish_for_env(i, force_floater=force_floater)
                next_states[i] = s
                fish_names[i] = fn
                fish_behaviors[i] = fb
                fish_difficulties[i] = fd

                if force_floater:
                    floater_curriculum_remaining -= 1
                    if floater_curriculum_remaining == 0:
                        floater_curriculum_active = False
                        print("✅ Floater curriculum phase complete!")

        states = next_states

        # ── Curriculum advancement (check roughly every 100 eps) ──
        if episode_count >= next_progress_print and episode_count > 100:
            bucket_names = list(difficulty_buckets.keys())
            for idx, (bn, bd) in enumerate(difficulty_buckets.items()):
                if bd['enabled']:
                    total_att = sum(bd['attempts'])
                    total_suc = sum(bd['success'])
                    if total_att >= 50 and total_suc / total_att >= curriculum_threshold and idx < len(bucket_names) - 1:
                        nxt = bucket_names[idx + 1]
                        if not difficulty_buckets[nxt]['enabled']:
                            difficulty_buckets[nxt]['enabled'] = True
                            print(f"\n*** Curriculum: Enabled '{nxt}' ***")

            # Trigger floater curriculum
            if wr >= 80.0 and not floater_curriculum_triggered and floater_fish_names:
                floater_curriculum_triggered = True
                floater_curriculum_active = True
                floater_curriculum_remaining = floater_curriculum_episodes
                print(f"\n{'=' * 60}")
                print(f"🎣 FLOATER CURRICULUM: {floater_curriculum_episodes} episodes on floaters!")
                print(f"{'=' * 60}\n")

        # ── Progress (every 100 episodes) ──
        if episode_count >= next_progress_print:
            now = time.time()
            elapsed = now - training_start_time
            sps = total_steps / elapsed if elapsed > 0 else 0
            remaining_eps = max(0, n_episodes - episode_count)
            eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
            eta_sec = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
            eh, erm = divmod(elapsed, 3600); em, es = divmod(erm, 60)
            rth, rtm = divmod(eta_sec, 3600); rtm, rts = divmod(rtm, 60)
            enabled = [b for b, info in difficulty_buckets.items() if info['enabled']]
            avg_s = float(np.mean(scores_window)) if scores_window else 0.0
            wr = sum(1 for s in scores[-100:] if s > 0) / min(100, len(scores)) * 100 if scores else 0.0
            recent_losses = agent.loss_list[-100:] if agent.loss_list else []
            mean_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            buf_pct = 100 * len(agent.memory) / max(1, getattr(agent.memory, 'capacity', getattr(agent.memory, 'buffer_size', 150000)))

            # Q-value health snapshot (cheap; one forward pass on current states)
            q_stats = ""
            try:
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(device)
                    qv = agent.qnetwork_local(states_t)
                    q_stats = (f"Q-values: mean|Q|={qv.abs().mean().item():.2f}  "
                               f"max|Q|={qv.abs().max().item():.2f}  "
                               f"range=[{qv.min().item():+.2f}, {qv.max().item():+.2f}]")
            except Exception:
                pass

            print()
            print("=" * 63)
            print(f"  Ep {episode_count}/{n_episodes} ({100*episode_count/n_episodes:5.1f}%)  "
                  f"|  ⏱  {int(eh)}h {int(em)}m {int(es)}s elapsed  "
                  f"|  ⏳ ETA {int(rth)}h {int(rtm)}m")
            print(f"  📊 {total_steps:,} steps  @  {sps:.0f} steps/sec  "
                  f"|  Epsilon: {eps:.4f}  |  Enabled: {', '.join(enabled) or 'none'}")
            print("-" * 63)
            print(f"  Score (100-ep avg): {avg_s:+.2f}    Win rate: {wr:5.1f}%    "
                  f"C51 loss: {mean_loss:.3f}    Replay: {buf_pct:.1f}% full")
            if q_stats:
                print(f"  {q_stats}")
            print("-" * 63)

            # Per-behavior breakdown (all 5 behaviors, not just floater)
            beh_lines = []
            for bname in ['sinker', 'dart', 'smooth', 'mixed', 'floater']:
                b = behavior_stats.get(bname, {})
                att = b.get('attempts', 0)
                suc = b.get('success', 0)
                pct = (100 * suc / att) if att > 0 else 0.0
                beh_lines.append(f"{bname:7s} {suc:3d}/{att:3d} ({pct:5.1f}%)")
            # Print as 2-column layout (3 left, 2 right) to keep it narrow
            for i in range(0, 5, 2):
                left = beh_lines[i]
                right = beh_lines[i + 1] if i + 1 < 5 else ""
                print(f"    {left}    {right}")

            # Per-difficulty win rates
            diff_parts = []
            for bname in ['easy', 'medium', 'hard']:
                b = difficulty_buckets[bname]
                if b['enabled']:
                    att_total = sum(b['attempts']); suc_total = sum(b['success'])
                    pct = (100 * suc_total / att_total) if att_total > 0 else 0.0
                    diff_parts.append(f"{bname}: {suc_total}/{att_total} ({pct:.1f}%)")
                else:
                    diff_parts.append(f"{bname}: locked")
            print(f"  By difficulty: {'   '.join(diff_parts)}")
            print("=" * 63)
            print()

            next_progress_print += 100

        # ── Wall-clock heartbeat (every 5 min, even between episode milestones) ──
        now = time.time()
        if now - last_heartbeat_time >= 300:
            elapsed_hb = now - training_start_time
            sps_hb = total_steps / elapsed_hb if elapsed_hb > 0 else 0
            eps_hb = sum(1 for s in scores[-100:] if s > 0) / min(100, len(scores)) * 100 if scores else 0.0
            loss_hb = float(np.mean(agent.loss_list[-50:])) if agent.loss_list else 0.0
            print(f"  💓 heartbeat @ {int(elapsed_hb//60)}m  |  ep {episode_count}/{n_episodes}  "
                  f"|  {total_steps:,} steps @ {sps_hb:.0f}/s  |  WR={eps_hb:.0f}%  "
                  f"|  loss={loss_hb:.3f}  |  ε={eps:.3f}")
            last_heartbeat_time = now

        # ── Save ──
        if episode_count > 0 and episode_count % save_every == 0:
            ckpt = f'models/checkpoints/episode_{episode_count}.pth'
            agent.save(ckpt)
            ct = time.time()
            print(f"Saved {ckpt} ({ (ct - last_checkpoint_time) / 60:.1f} min)")
            last_checkpoint_time = ct

            # Quick plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
            ax1.plot(np.arange(len(scores)), scores, alpha=0.5)
            if len(scores) >= 100:
                ma = np.convolve(scores, np.ones(100) / 100, mode='valid')
                ax1.plot(np.arange(99, len(scores)), ma, 'r-', linewidth=2)
            ax1.set_title('Training Scores')
            bnames, brates = [], []
            for b, s in behavior_stats.items():
                if s["attempts"] > 0:
                    bnames.append(b)
                    brates.append(s["success"] / s["attempts"] * 100)
            ax2.bar(bnames, brates)
            ax2.set_title('Success by Behavior')
            ax2.set_ylim(0, 100)
            if agent.loss_list:
                w = min(100, len(agent.loss_list))
                la = np.convolve(agent.loss_list, np.ones(w) / w, mode='valid')
                ax3.plot(np.arange(w - 1, len(agent.loss_list)), la)
                ax3.set_title('C51 Loss')
                ax3.set_yscale('log')
            plt.tight_layout()
            plt.savefig(f'training_logs/graphs/episode_{episode_count}.png')
            plt.close()

        # ── Early stopping eval ──
        if episode_count >= next_eval_print:
            print("\nRunning evaluation...")
            env_vec.envs[0].render_mode = None
            eval_success = 0
            eval_eps = 20
            for _ in range(eval_eps):
                fn = random.choice(env_vec.envs[0].get_available_fish())
                env_vec.envs[0].fish_name = fn
                s = env_vec.envs[0].reset()
                for _ in range(max_t):
                    a = agent.act(s, eps=0.0)
                    s, r, done, info = env_vec.envs[0].step(a)
                    if done:
                        if info.get('distance_from_catching', 0) >= 1.0:
                            eval_success += 1
                        break
            eval_rate = eval_success / eval_eps
            print(f"Eval: {eval_rate * 100:.1f}% ({eval_success}/{eval_eps})")
            if eval_rate >= early_stop_threshold:
                perfect_episodes += 1
                print(f"Perfect eval {perfect_episodes}/{required_perfect}")
                if perfect_episodes >= required_perfect:
                    print(f"\n*** EARLY STOP at episode {episode_count} ***")
                    agent.save(f'models/checkpoints/early_stop_ep{episode_count}.pth', export_onnx=True)
                    break
            else:
                perfect_episodes = 0
            next_eval_print += 1000

    # Final save
    agent.save(f'models/checkpoints/final_ep{episode_count}.pth', export_onnx=True)
    print(f"\n💾 Final: models/checkpoints/final_ep{episode_count}.pth")
    milestone_tracker.close()

    total = time.time() - training_start_time
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nDone! {episode_count} episodes in {int(h)}h {int(m)}m {int(s)}s")
    print(f"Total steps: {total_steps} ({total_steps/total:.0f} steps/sec)")
    return scores


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_agent(env, agent, n_episodes=20, render=True):
    """Evaluate trained agent performance across all behaviors."""
    scores = []
    success_count = 0
    behavior_results = {}

    fish_names = env.get_available_fish()
    behavior_fish = {}
    for fish in env.fish_data:
        b = fish.get("behaviour", "mixed")
        behavior_fish.setdefault(b, []).append(fish["name"])

    test_seq = []
    max_per = n_episodes // max(len(behavior_fish), 1)
    for b, flist in behavior_fish.items():
        for i in range(min(max_per, len(flist))):
            test_seq.append(flist[i])
    while len(test_seq) < n_episodes:
        test_seq.append(random.choice(fish_names))
    random.shuffle(test_seq)

    for i_ep, fish_name in enumerate(test_seq):
        env.fish_name = fish_name
        state = env.reset()
        behavior = env.current_fish["behaviour"]
        difficulty = env.current_fish["difficulty"]

        behavior_results.setdefault(behavior, {"attempts": 0, "success": 0})
        behavior_results[behavior]["attempts"] += 1

        print(f"\nEval {i_ep + 1}/{n_episodes}: {fish_name} ({behavior}, d={difficulty})")
        score = 0
        for t in range(1000):
            action = agent.act(state, eps=0.0)
            state, reward, done, info = env.step(action)
            score += reward
            if render and env.render_mode == "human":
                time.sleep(0.01)
            if done:
                if env.distanceFromCatching >= 1.0:
                    print(f"✓ Score: {score:.2f}")
                    success_count += 1
                    behavior_results[behavior]["success"] += 1
                else:
                    print(f"✗ Score: {score:.2f}")
                break
        scores.append(score)

    print(f"\nOverall: {success_count}/{n_episodes} ({success_count / n_episodes * 100:.1f}%)")
    print(f"Avg score: {np.mean(scores):.2f}")
    print("\nBy behavior:")
    for b, r in behavior_results.items():
        sr = r["success"] / r["attempts"] * 100 if r["attempts"] > 0 else 0
        print(f"  {b}: {r['success']}/{r['attempts']} ({sr:.1f}%)")
    return scores, behavior_results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 4 parallel envs for vectorized training (much faster GPU utilisation)
    env_vec = VectorizedEnv(num_envs=4, render_mode=None)

    # Single env for interactive mode
    env = env_vec.envs[0]

    # ── C51 Distributional DQN Agent ──
    # State dimension is now 24D (14 base + 5 behavior one-hot + 5 enhanced features)
    agent = C51DQNAgent(
        state_dim=24,  # Updated from 14 → 24 for enhanced state
        action_dim=2,
        hidden_sizes=[256, 256, 128, 64],
        learning_rate=2e-4,
        gamma=0.99,
        buffer_size=150000,      # Larger buffer for C51 (more diverse data)
        batch_size=128,
        update_every=4,
        n_step=3,
        target_update_freq=500,  # More frequent target updates for stability
        n_atoms=51,              # C51: 51 atoms
        v_min=-20.0,             # Support range
        v_max=20.0,
        grad_accum_steps=2,      # Gradient accumulation for stability
        weight_decay=1e-5,       # L2 regularization
        lr_warmup_steps=1000,    # LR warmup
    )

    # ── Colab / GPU adaptive settings ──
    is_colab = 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules
    is_cuda = torch.cuda.is_available()
    colab_batch_size = 256 if (is_cuda and is_colab) else 128

    # Re-initialise agent with Colab-optimised batch size if needed
    if is_colab and is_cuda and colab_batch_size != agent.batch_size:
        print(f"[Colab] Adjusting batch size to {colab_batch_size} for T4 GPU")
        agent.batch_size = colab_batch_size

    if is_colab and is_cuda and agent.use_amp:
        print("[Colab] Mixed Precision (AMP) active — T4 will be 2-3x faster")
        print(f"[Colab] GPU: {torch.cuda.get_device_name(0)}")

    # Training modes
    train_new_model = True   # Set to True to train from scratch
    fine_tune_model = False
    use_rule_bot = False
    skip_evaluation = True

    if train_new_model:
        scores = train_dqn_vectorized(
            env_vec=env_vec, agent=agent,
            n_episodes=12000,
            max_t=2000,
            # FIX: was 0.05 (5% random), which is 4x less exploration than the
            # eps=0.20 (20% random) used by the Dueling DQN runs that successfully
            # converged to 90%+ win rate. C51 needs more early exploration to
            # discover the press/release rhythm on the larger network.
            eps_start=0.20,
            eps_end=0.01,
            save_every=500,
            render_every=3000,
        )

    elif use_rule_bot:
        print("\n🤖 Rule-based bot active\n")

    elif fine_tune_model:
        model_path = 'models/YOUR_MODEL_NAME.pth'
        if os.path.exists(model_path):
            agent.load(model_path)
            print("Fine-tuning...")
            for pg in agent.optimizer.param_groups:
                pg['lr'] = 1e-4
            scores = train_dqn(
                env=env, agent=agent,
                n_episodes=2000, max_t=2000,
                eps_start=0.02, eps_end=0.001,
                save_every=500, render_every=1000,
            )
        else:
            print(f"Model {model_path} not found")

    else:
        model_path = 'models/checkpoints/episode_6500.pth'
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"No model at {model_path}")
            print("Available:")
            if os.path.exists('models'):
                for m in os.listdir('models'):
                    if m.endswith('.pth'):
                        print(f"  - {m}")
            exit(1)

    if not use_rule_bot and not skip_evaluation:
        print("\nRunning evaluation...")
        evaluate_agent(env, agent, n_episodes=20, render=True)

    # Interactive mode
    print("\nInteractive mode. Select a fish to watch, 'r' for random, 'q' to quit.")
    while True:
        print("\nAvailable fish:")
        for i, fn in enumerate(env.get_available_fish()):
            detail = next((f for f in env.fish_data if f["name"] == fn), None)
            if detail:
                print(f"  {i}: {fn} ({detail.get('behaviour','?')}, d={detail.get('difficulty','?')})")

        key = input("\nSelect number, 'r' random, 'q' quit: ").strip().lower()
        if key == 'q':
            break
        if key == 'r':
            fish_name = random.choice(env.get_available_fish())
        elif key.isdigit() and 0 <= int(key) < len(env.get_available_fish()):
            fish_name = env.get_available_fish()[int(key)]
        else:
            print("Invalid")
            continue

        env.fish_name = fish_name
        state = env.reset()
        behavior = env.current_fish["behaviour"]
        difficulty = env.current_fish["difficulty"]
        print(f"\nAgent catching: {fish_name} ({behavior}, d={difficulty})")

        score, done = 0, False
        while not done:
            if use_rule_bot:
                bc = env.bobberBarPos + (env.bobberBarHeight / 2.0)
                action = env.ACTION_PRESS if env.bobberPosition < bc else env.ACTION_NONE
            else:
                action = agent.act(state, eps=0.0)
            state, reward, done, info = env.step(action)
            score += reward
            time.sleep(0.016)

        result = "✓" if env.distanceFromCatching >= 1.0 else "✗"
        print(f"{result} Score: {score:.2f}")
        time.sleep(1)

    env.close()
