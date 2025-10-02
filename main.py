import csv
import os
import random
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

# Set up device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the experience tuple structure
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class VectorizedEnv:
    """Run multiple fishing environments in parallel for faster data collection"""

    def __init__(self, num_envs=4, **env_kwargs):
        """
        Args:
            num_envs: Number of parallel environments
            **env_kwargs: Arguments to pass to each environment
        """
        self.num_envs = num_envs
        # Disable rendering for vectorized envs
        env_kwargs['render_mode'] = None
        self.envs = [FishingMinigameEnv(**env_kwargs) for _ in range(num_envs)]
        self.dones = [False] * num_envs

    def reset(self):
        """Reset all environments"""
        states = [env.reset() for env in self.envs]
        self.dones = [False] * self.num_envs
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        """Step all environments with given actions"""
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                # Environment already done, reset it
                state = env.reset()
                results.append((state, 0.0, False, {}))
            else:
                results.append(env.step(action))
                self.dones[i] = results[-1][2]  # Update done status

        states, rewards, dones, infos = zip(*results)
        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos)
        )

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


class MilestoneTracker:
    """Track and log major training milestones for YouTube video storytelling"""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Milestone tracking
        self.milestones = {
            # First success
            'first_success': False,

            # Win streaks
            'first_win_streak_3': False,
            'first_win_streak_5': False,
            'first_win_streak_10': False,
            'first_win_streak_25': False,
            'first_win_streak_50': False,
            'first_win_streak_100': False,

            # Difficulty mastery
            'easy_mastery_75': False,
            'easy_mastery_90': False,
            'easy_mastery_95': False,
            'medium_unlocked': False,
            'medium_mastery_75': False,
            'medium_mastery_90': False,
            'hard_unlocked': False,
            'hard_mastery_75': False,
            'hard_mastery_90': False,

            # Overall performance tiers
            'overall_50_percent': False,
            'overall_75_percent': False,
            'overall_80_percent': False,
            'overall_90_percent': False,
            'overall_95_percent': False,
            'overall_99_percent': False,

            # Behavior type discoveries
            'first_sinker_catch': False,
            'first_dart_catch': False,
            'first_smooth_catch': False,
            'first_mixed_catch': False,
            'first_floater_catch': False,

            # Behavior mastery
            'sinker_mastery_80': False,
            'dart_mastery_80': False,
            'smooth_mastery_80': False,
            'mixed_mastery_80': False,
            'floater_mastery_80': False,

            # Epsilon milestones (learning progress)
            'epsilon_below_0_5': False,
            'epsilon_below_0_25': False,
            'epsilon_below_0_1': False,
            'epsilon_below_0_01': False,

            # Episode milestones
            'episode_100': False,
            'episode_500': False,
            'episode_1000': False,
            'episode_2500': False,
            'episode_5000': False,

            # Special achievements
            'first_perfect_eval': False,  # 10/10 in evaluation
            'first_flawless_20': False,   # 20+ win streak
            'speed_demon_50': False,      # Catch in under 50 steps
            'comeback_after_5_losses': False,
            'all_behaviors_caught': False,  # Caught all 5 behavior types
        }

        # Tracking variables
        self.current_win_streak = 0
        self.max_win_streak = 0
        self.episode_results = []  # For CSV logging
        self.consecutive_losses = 0

        # Behavior tracking
        self.behavior_stats = {
            'sinker': {'attempts': 0, 'successes': 0},
            'dart': {'attempts': 0, 'successes': 0},
            'smooth': {'attempts': 0, 'successes': 0},
            'mixed': {'attempts': 0, 'successes': 0},
            'floater': {'attempts': 0, 'successes': 0},
        }
        self.behaviors_caught = set()  # Track which behaviors have been caught

        # Speed tracking
        self.shortest_catch = float('inf')

        # CSV setup
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

        # Milestone log file
        self.milestone_log_path = os.path.join(log_dir, f"milestones_{timestamp}.txt")
        with open(self.milestone_log_path, 'w') as f:
            f.write(f"Training Milestones Log - Started {timestamp}\n")
            f.write("=" * 60 + "\n\n")

    def check_milestone(self, name, condition, episode, message):
        """Check and log a milestone if not already achieved"""
        if not self.milestones.get(name, False) and condition:
            self.milestones[name] = True
            log_msg = f"🏆 MILESTONE at Episode {episode}: {message}"
            print(f"\n{'='*60}")
            print(log_msg)
            print(f"{'='*60}\n")

            # Write to milestone log
            with open(self.milestone_log_path, 'a') as f:
                f.write(f"Episode {episode}: {message}\n")

            return True
        return False

    def update(self, episode, score, success, fish_info, epsilon, stats):
        """Update tracking with episode results"""
        # Update behavior stats
        behavior = fish_info['behavior']
        if behavior in self.behavior_stats:
            self.behavior_stats[behavior]['attempts'] += 1
            if success:
                self.behavior_stats[behavior]['successes'] += 1
                self.behaviors_caught.add(behavior)

        # Update speed tracking
        if success and fish_info['episode_length'] < self.shortest_catch:
            self.shortest_catch = fish_info['episode_length']

        # Update win streak and loss tracking
        if success:
            self.current_win_streak += 1
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
            self.consecutive_losses = 0
        else:
            self.current_win_streak = 0
            self.consecutive_losses += 1

        # Calculate behavior success rates
        behavior_rates = {}
        for behavior_type, data in self.behavior_stats.items():
            if data['attempts'] > 0:
                behavior_rates[behavior_type] = (data['successes'] / data['attempts']) * 100
            else:
                behavior_rates[behavior_type] = 0.0

        # Write to CSV
        self.csv_writer.writerow([
            episode,
            f"{score:.2f}",
            1 if success else 0,
            fish_info['name'],
            fish_info['difficulty'],
            fish_info['behavior'],
            fish_info['episode_length'],
            f"{epsilon:.4f}",
            self.current_win_streak,
            f"{stats['avg_score']:.2f}",
            f"{stats['win_rate']:.1f}",
            f"{stats['easy_success_rate']:.1f}",
            f"{stats['medium_success_rate']:.1f}",
            f"{stats['hard_success_rate']:.1f}",
            self.shortest_catch if self.shortest_catch != float('inf') else 0,
            len(self.behaviors_caught),
            f"{behavior_rates.get('sinker', 0):.1f}",
            f"{behavior_rates.get('dart', 0):.1f}",
            f"{behavior_rates.get('smooth', 0):.1f}",
            f"{behavior_rates.get('mixed', 0):.1f}",
            f"{behavior_rates.get('floater', 0):.1f}",
        ])
        self.csv_file.flush()  # Ensure data is written immediately

        # === MILESTONE CHECKS ===

        # Episode milestones
        if episode == 100:
            self.check_milestone('episode_100', True, episode, "Reached episode 100!")
        elif episode == 500:
            self.check_milestone('episode_500', True, episode, "Reached episode 500!")
        elif episode == 1000:
            self.check_milestone('episode_1000', True, episode, "Reached episode 1000!")
        elif episode == 2500:
            self.check_milestone('episode_2500', True, episode, "Reached episode 2500!")
        elif episode == 5000:
            self.check_milestone('episode_5000', True, episode, "Reached episode 5000!")

        # First success
        if success:
            self.check_milestone('first_success', True, episode,
                               f"First successful catch! ({fish_info['name']})")

        # Win streak milestones
        if self.current_win_streak == 3:
            self.check_milestone('first_win_streak_3', True, episode,
                               "First 3-episode win streak!")
        elif self.current_win_streak == 5:
            self.check_milestone('first_win_streak_5', True, episode,
                               "First 5-episode win streak!")
        elif self.current_win_streak == 10:
            self.check_milestone('first_win_streak_10', True, episode,
                               "First 10-episode win streak!")
        elif self.current_win_streak == 25:
            self.check_milestone('first_win_streak_25', True, episode,
                               "First 25-episode win streak! 🔥")
        elif self.current_win_streak == 50:
            self.check_milestone('first_win_streak_50', True, episode,
                               "First 50-episode win streak! Incredible! 🔥🔥")
        elif self.current_win_streak == 100:
            self.check_milestone('first_win_streak_100', True, episode,
                               "First 100-episode win streak! LEGENDARY! 🔥🔥🔥")

        # Flawless streak (20+ is special)
        if self.current_win_streak >= 20:
            self.check_milestone('first_flawless_20', True, episode,
                               f"Flawless performance! {self.current_win_streak}-episode win streak!")

        # Behavior discoveries
        if success:
            if behavior == 'sinker':
                self.check_milestone('first_sinker_catch', True, episode,
                                   f"First sinker fish caught! ({fish_info['name']})")
            elif behavior == 'dart':
                self.check_milestone('first_dart_catch', True, episode,
                                   f"First dart fish caught! ({fish_info['name']})")
            elif behavior == 'smooth':
                self.check_milestone('first_smooth_catch', True, episode,
                                   f"First smooth fish caught! ({fish_info['name']})")
            elif behavior == 'mixed':
                self.check_milestone('first_mixed_catch', True, episode,
                                   f"First mixed behavior fish caught! ({fish_info['name']})")
            elif behavior == 'floater':
                self.check_milestone('first_floater_catch', True, episode,
                                   f"First floater fish caught! ({fish_info['name']})")

        # All behaviors caught
        if len(self.behaviors_caught) >= 5:
            self.check_milestone('all_behaviors_caught', True, episode,
                               "Caught all 5 fish behavior types!")

        # Behavior mastery (80%+ success rate, minimum 20 attempts)
        for behavior_type, rate in behavior_rates.items():
            if self.behavior_stats[behavior_type]['attempts'] >= 20 and rate >= 80:
                milestone_key = f'{behavior_type}_mastery_80'
                self.check_milestone(milestone_key, True, episode,
                                   f"{behavior_type.capitalize()} fish mastered! ({rate:.1f}% success rate)")

        # Speed achievements
        if success and fish_info['episode_length'] < 50:
            self.check_milestone('speed_demon_50', True, episode,
                               f"Speed demon! Caught in {fish_info['episode_length']} steps! ({fish_info['name']})")

        # Comeback achievement (success after 5+ consecutive losses)
        if success and self.consecutive_losses >= 5:
            self.check_milestone('comeback_after_5_losses', True, episode,
                               f"Epic comeback! Won after {self.consecutive_losses} consecutive losses!")

        # Epsilon milestones (learning progress)
        if epsilon < 0.5:
            self.check_milestone('epsilon_below_0_5', True, episode,
                               f"Exploration → Exploitation: Epsilon dropped below 0.5 ({epsilon:.4f})")
        if epsilon < 0.25:
            self.check_milestone('epsilon_below_0_25', True, episode,
                               f"Mostly exploiting learned policy: Epsilon below 0.25 ({epsilon:.4f})")
        if epsilon < 0.1:
            self.check_milestone('epsilon_below_0_1', True, episode,
                               f"Expert mode: Epsilon below 0.1 ({epsilon:.4f})")
        if epsilon < 0.01:
            self.check_milestone('epsilon_below_0_01', True, episode,
                               f"Pure exploitation: Epsilon below 0.01 ({epsilon:.4f})")

        # Difficulty mastery milestones
        if stats['easy_success_rate'] >= 75:
            self.check_milestone('easy_mastery_75', True, episode,
                               f"Easy fish mastery! ({stats['easy_success_rate']:.1f}% success rate)")
        if stats['easy_success_rate'] >= 90:
            self.check_milestone('easy_mastery_90', True, episode,
                               f"Easy fish excellence! ({stats['easy_success_rate']:.1f}% success rate)")
        if stats['easy_success_rate'] >= 95:
            self.check_milestone('easy_mastery_95', True, episode,
                               f"Easy fish domination! ({stats['easy_success_rate']:.1f}% success rate)")

        if stats['medium_enabled']:
            self.check_milestone('medium_unlocked', True, episode,
                               "Medium difficulty unlocked!")
            if stats['medium_success_rate'] >= 75:
                self.check_milestone('medium_mastery_75', True, episode,
                                   f"Medium fish mastery! ({stats['medium_success_rate']:.1f}% success rate)")
            if stats['medium_success_rate'] >= 90:
                self.check_milestone('medium_mastery_90', True, episode,
                                   f"Medium fish excellence! ({stats['medium_success_rate']:.1f}% success rate)")

        if stats['hard_enabled']:
            self.check_milestone('hard_unlocked', True, episode,
                               "Hard difficulty unlocked!")
            if stats['hard_success_rate'] >= 75:
                self.check_milestone('hard_mastery_75', True, episode,
                                   f"Hard fish mastery! ({stats['hard_success_rate']:.1f}% success rate)")
            if stats['hard_success_rate'] >= 90:
                self.check_milestone('hard_mastery_90', True, episode,
                                   f"Hard fish excellence! ({stats['hard_success_rate']:.1f}% success rate)")

        # Overall performance milestones
        if stats['win_rate'] >= 50:
            self.check_milestone('overall_50_percent', True, episode,
                               f"Breaking even! 50% win rate achieved! ({stats['win_rate']:.1f}%)")
        if stats['win_rate'] >= 75:
            self.check_milestone('overall_75_percent', True, episode,
                               f"Strong performance! 75% win rate! ({stats['win_rate']:.1f}%)")
        if stats['win_rate'] >= 80:
            self.check_milestone('overall_80_percent', True, episode,
                               f"Expert level! 80% win rate! ({stats['win_rate']:.1f}%)")
        if stats['win_rate'] >= 90:
            self.check_milestone('overall_90_percent', True, episode,
                               f"Master angler! 90% win rate! ({stats['win_rate']:.1f}%)")
        if stats['win_rate'] >= 95:
            self.check_milestone('overall_95_percent', True, episode,
                               f"Elite performance! 95% win rate! ({stats['win_rate']:.1f}%)")
        if stats['win_rate'] >= 99:
            self.check_milestone('overall_99_percent', True, episode,
                               f"GODLIKE! 99% win rate! ({stats['win_rate']:.1f}%)")

        # Perfect evaluation (check if stats contain eval info)
        if 'eval_success_rate' in stats and stats['eval_success_rate'] >= 100:
            self.check_milestone('first_perfect_eval', True, episode,
                               "Perfect evaluation! 10/10 catches!")

    def close(self):
        """Close CSV file"""
        self.csv_file.close()
        print(f"\n📊 Training metrics saved to: {self.csv_path}")
        print(f"🏆 Milestone log saved to: {self.milestone_log_path}")


class SumTree:
    """Sum tree data structure for prioritized replay buffer"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Update tree with priority change"""
        parent = (idx - 1) // 2
        self.tree[parent] += float(change)
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return sum of all priorities"""
        return self.tree[0]

    def add(self, priority, data):
        """Store priority and sample"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Update priority"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get priority and sample"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with optimized tensor operations"""

    epsilon = 1e-5  # Small constant to avoid zero priority
    alpha = 0.6  # Priority exponent
    beta = 0.4  # Importance sampling weight
    beta_increment = 0.001  # Annealing rate
    abs_err_upper = 1.0  # Clipped abs error

    def __init__(self, capacity=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.use_pinned_memory = torch.cuda.is_available()

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        experience = Experience(state, action, reward, next_state, done)
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        """Sample a batch of experiences with priorities"""
        experiences = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        # Anneal beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            experiences.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Pre-allocate arrays for better performance
        state_dim = experiences[0].state.shape[0]
        states = np.zeros((batch_size, state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, 1), dtype=np.int64)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        next_states = np.zeros((batch_size, state_dim), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        # Fill arrays
        for idx, e in enumerate(experiences):
            states[idx] = e.state
            actions[idx] = e.action
            rewards[idx] = e.reward
            next_states[idx] = e.next_state
            dones[idx] = e.done

        # Convert to tensors with optimized transfer
        if self.use_pinned_memory:
            states = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
            actions = torch.from_numpy(actions).pin_memory().to(device, non_blocking=True)
            rewards = torch.from_numpy(rewards).pin_memory().to(device, non_blocking=True)
            next_states = torch.from_numpy(next_states).pin_memory().to(device, non_blocking=True)
            dones = torch.from_numpy(dones).pin_memory().to(device, non_blocking=True)
        else:
            states = torch.from_numpy(states).to(device)
            actions = torch.from_numpy(actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            dones = torch.from_numpy(dones).to(device)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights.astype(np.float32)).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones, indices, is_weights

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)

    def __len__(self):
        """Return current size of buffer"""
        return self.tree.n_entries


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network with Layer Normalization and Dropout"""

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=[128, 128, 64], dropout_rate=0.2):
        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim

        # Shared feature extraction layers with Layer Normalization and Dropout
        feature_layers = []
        input_size = state_dim

        for i, hidden_size in enumerate(hidden_sizes):
            feature_layers.append(nn.Linear(input_size, hidden_size))
            feature_layers.append(nn.LayerNorm(hidden_size))
            feature_layers.append(nn.ReLU())
            # Add dropout after ReLU (but not on last layer)
            if i < len(hidden_sizes) - 1:
                feature_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.feature_layer = nn.Sequential(*feature_layers)

        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for ReLU networks
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        """Forward pass through the dueling architecture
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages using the dueling architecture formula
        # Subtract mean advantage to ensure identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DQNAgent:
    """Agent implementing Double DQN with Dueling architecture, Prioritized Replay, and N-step returns"""

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=None, learning_rate=3e-4, gamma=0.99,
                 buffer_size=100000, batch_size=128, update_every=4, n_step=3, target_update_freq=1000):
        """Initialize agent parameters and build models"""
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 64]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.n_step = n_step  # N-step returns
        self.target_update_freq = target_update_freq  # Hard update frequency

        # Q-Networks with Dueling architecture
        self.qnetwork_local = DuelingDQN(state_dim, action_dim, hidden_sizes).to(device)
        self.qnetwork_target = DuelingDQN(state_dim, action_dim, hidden_sizes).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()  # Target network always in eval mode

        # PyTorch 2.0+ compile optimization (if available)
        # Note: Disabled due to compatibility issues with Dropout layers causing access violations
        # if hasattr(torch, 'compile'):
        #     try:
        #         self.qnetwork_local = torch.compile(self.qnetwork_local)
        #         self.qnetwork_target = torch.compile(self.qnetwork_target)
        #         print("✅ PyTorch 2.0 compile optimization enabled")
        #     except Exception as e:
        #         print(f"⚠️  Could not enable torch.compile: {e}")

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-5)

        # Mixed Precision Training (AMP) for speed optimization
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("✅ Mixed Precision Training (AMP) enabled")

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)

        # N-step buffer for multi-step returns
        self.n_step_buffer = deque(maxlen=n_step)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = update_every
        self.total_steps = 0

        # For tracking statistics
        self.loss_list = []

    def step(self, state, action, reward, next_state, done):
        """Save experience with N-step returns in replay memory, and use random sample to learn"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we have enough steps or episode ended, compute n-step return and add to memory
        if len(self.n_step_buffer) == self.n_step or done:
            # Compute n-step return
            n_step_state = self.n_step_buffer[0][0]
            n_step_action = self.n_step_buffer[0][1]
            n_step_reward = sum([self.gamma ** i * exp[2] for i, exp in enumerate(self.n_step_buffer)])
            n_step_next_state = self.n_step_buffer[-1][3]
            n_step_done = self.n_step_buffer[-1][4]

            # Add to memory
            self.memory.add(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)

            # If done, flush remaining experiences in n-step buffer
            if done:
                self.n_step_buffer.clear()

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        self.total_steps += 1

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                loss = self.learn(experiences)
                self.loss_list.append(loss)

                # Hard update target network periodically
                if self.total_steps % self.target_update_freq == 0:
                    self.hard_update()

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy

        Args:
            state: current state
            eps: epsilon for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # No need for eval/train switching - torch.no_grad() is sufficient
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def learn(self, experiences):
        """Update value parameters using batch of experience tuples with Double DQN and Prioritized Replay

        Args:
            experiences: tuple of (s, a, r, s', done, indices, is_weights) tuples
        """
        states, actions, rewards, next_states, dones, indices, is_weights = experiences

        # Double DQN: use local network to select actions, target network to evaluate
        with torch.no_grad():
            # Get action indices from local model (best actions)
            action_indices = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)

            # Get Q values from target model for those actions
            Q_targets_next = self.qnetwork_target(next_states).gather(1, action_indices)

            # Compute Q targets for current states using Bellman equation with n-step adjustment
            Q_targets = rewards + (self.gamma ** self.n_step * Q_targets_next * (1 - dones))

        # Forward pass with Mixed Precision if available
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                # Get expected Q values from local model
                Q_expected = self.qnetwork_local(states).gather(1, actions)

                # Calculate weighted loss (importance sampling)
                loss = (is_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()

            # Calculate TD errors for priority updates (outside autocast)
            with torch.no_grad():
                td_errors = (Q_expected.float() - Q_targets).detach().cpu().numpy()

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training (CPU or older GPUs)
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Calculate TD errors for priority updates
            td_errors = (Q_expected - Q_targets).detach().cpu().numpy()

            # Calculate weighted loss (importance sampling)
            loss = (is_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)

            self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Update priorities in the replay buffer
        self.memory.update_priorities(indices, td_errors)

        # Return loss value for monitoring
        return loss.item()

    def hard_update(self):
        """Hard update: copy weights from local to target network"""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        print(f"Target network updated at step {self.total_steps}")

    def save(self, filename, export_onnx=False, state_dim=14):
        """Save trained model in PyTorch format and optionally ONNX

        Args:
            filename: Path to save .pth file (e.g., 'models/my_model.pth')
            export_onnx: Also save as ONNX with same base name
            state_dim: State dimension for ONNX export (default 14)
        """
        # Save PyTorch checkpoint
        torch.save({'local_state_dict': self.qnetwork_local.state_dict(),
                    'target_state_dict': self.qnetwork_target.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss_list': self.loss_list}, filename)

        # Also export to ONNX if requested
        if export_onnx:
            # Replace .pth extension with .onnx
            onnx_filename = filename.replace('.pth', '.onnx')
            self.export_onnx(onnx_filename, state_dim=state_dim)

    def export_onnx(self, filename, state_dim=14):
        """Export model to ONNX format for deployment

        Args:
            filename: Path to save ONNX file (e.g., 'models/model.onnx')
            state_dim: State dimension (default 14 for current setup)
        """
        # Set model to eval mode
        self.qnetwork_local.eval()

        # Create dummy input (batch_size=1, state_dim)
        dummy_input = torch.randn(1, state_dim, device=device)

        # Export to ONNX
        torch.onnx.export(
            self.qnetwork_local,
            dummy_input,
            filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['q_values'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'q_values': {0: 'batch_size'}
            }
        )
        print(f"✅ Model exported to ONNX: {filename}")

        # Set back to train mode if needed
        self.qnetwork_local.train()

    def load(self, filename):
        """Load trained model"""
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))

        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint.get('loss_list', [])

        print(f"Model loaded from {filename}")


def train_dqn(env, agent, n_episodes=10000, max_t=2000, eps_start=0.2, eps_end=0.001,
              save_every=500, render_every=1000):
    """Train DQN agent with adaptive curriculum learning and improved exploration

    Args:
        env: environment
        agent: DQN agent
        n_episodes: maximum number of training episodes
        max_t: maximum number of timesteps per episode
        eps_start: starting value of epsilon for epsilon-greedy action selection
        eps_end: minimum value of epsilon
        save_every: how often to save the model (episodes)
        render_every: how often to render an episode
    """
    scores = []  # list of scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for tracking progress

    # Create directories for saving models and graphs
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_logs/graphs", exist_ok=True)

    # Initialize milestone tracker for YouTube video
    milestone_tracker = MilestoneTracker(log_dir="training_logs")
    print(f"📊 Logging detailed metrics to: {milestone_tracker.csv_path}")
    print(f"🏆 Milestone tracking enabled\n")

    # Track difficulties mastered with adaptive thresholds
    difficulty_buckets = {
        'easy': {'range': (0, 40), 'attempts': 0, 'success': 0, 'enabled': True},
        'medium': {'range': (41, 70), 'attempts': 0, 'success': 0, 'enabled': False},
        'hard': {'range': (71, 100), 'attempts': 0, 'success': 0, 'enabled': False}
    }

    # For plotting
    fish_behaviors = list(env.BEHAVIOR_TYPES.keys())
    behavior_stats = {b: {"attempts": 0, "success": 0} for b in fish_behaviors}

    # Enhanced curriculum learning with adaptive difficulty mixing
    curriculum_threshold = 0.75  # 75% success rate to enable next difficulty
    difficulty_mix_ratio = 0.8  # 80% current level, 20% harder when mixing

    # For early stopping
    perfect_episodes = 0
    required_perfect = 3  # Number of consecutive evaluation rounds with near-perfect performance
    early_stop_threshold = 0.98  # 98% success rate

    # For logging purposes
    training_start_time = time.time()
    last_checkpoint_time = training_start_time

    print(f"Starting training with up to {n_episodes} episodes...")
    print(
        f"Early stopping after {required_perfect} consecutive evaluations with >{early_stop_threshold * 100}% success rate")

    for i_episode in range(1, n_episodes + 1):
        # Cosine annealing epsilon schedule (smoother decay)
        eps = eps_end + (eps_start - eps_end) * (1 + np.cos(np.pi * i_episode / n_episodes)) / 2

        # Adaptive curriculum learning - select fish based on enabled difficulty buckets
        enabled_buckets = [b for b, info in difficulty_buckets.items() if info['enabled']]

        # Mix difficulties: prefer current level but include some harder fish
        if len(enabled_buckets) > 1 and random.random() > difficulty_mix_ratio:
            # Sample from harder difficulties
            bucket = enabled_buckets[-1]
        else:
            # Sample from current/earlier difficulties
            bucket = random.choice(enabled_buckets[:-1] if len(enabled_buckets) > 1 else enabled_buckets)

        bucket_info = difficulty_buckets[bucket]
        min_diff, max_diff = bucket_info['range']

        # Get fish in this difficulty range
        available_fish = [f for f in env.fish_data
                          if min_diff <= f["difficulty"] <= max_diff]

        if not available_fish:  # Fallback if filter gives no fish
            available_fish = env.fish_data

        # Reset environment with appropriate fish
        fish_name = random.choice([f["name"] for f in available_fish])
        env.fish_name = fish_name
        state = env.reset()

        # Get fish details
        fish_behavior = env.current_fish["behaviour"]
        fish_difficulty = env.current_fish["difficulty"]
        behavior_stats[fish_behavior]["attempts"] += 1

        # Track which bucket this fish belongs to
        for bucket_name, bucket_data in difficulty_buckets.items():
            b_min, b_max = bucket_data['range']
            if b_min <= fish_difficulty <= b_max:
                bucket_data['attempts'] += 1
                break

        score = 0
        render = (i_episode % render_every == 0)
        render_mode_backup = env.render_mode

        if render:
            env.render_mode = "human"
            print(f"\nEpisode {i_episode}: Rendering... Fish: {fish_name} ({fish_behavior})")
        else:
            env.render_mode = None

        # Run episode
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if render:
                time.sleep(0.01)  # slow down rendering

            if done:
                if env.distanceFromCatching >= 1.0:  # Successfully caught fish
                    behavior_stats[fish_behavior]["success"] += 1

                    # Track success for difficulty bucket
                    for bucket_name, bucket_data in difficulty_buckets.items():
                        b_min, b_max = bucket_data['range']
                        if b_min <= fish_difficulty <= b_max:
                            bucket_data['success'] += 1
                            break

                break

        # Restore render mode
        env.render_mode = render_mode_backup

        # Record score
        scores_window.append(score)
        scores.append(score)

        # Calculate stats for milestone tracking
        avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0.0
        success_count = sum(1 for i in range(max(0, len(scores) - 100), len(scores)) if scores[i] > 0)
        win_rate = success_count / min(100, len(scores)) * 100

        # Calculate per-difficulty success rates
        def calc_bucket_rate(bucket_name):
            bucket = difficulty_buckets[bucket_name]
            if bucket['attempts'] > 0:
                return (bucket['success'] / bucket['attempts']) * 100
            return 0.0

        # Update milestone tracker
        milestone_tracker.update(
            episode=i_episode,
            score=score,
            success=(env.distanceFromCatching >= 1.0),
            fish_info={
                'name': fish_name,
                'difficulty': fish_difficulty,
                'behavior': fish_behavior,
                'episode_length': env.episode_length
            },
            epsilon=eps,
            stats={
                'avg_score': avg_score,
                'win_rate': win_rate,
                'easy_success_rate': calc_bucket_rate('easy'),
                'medium_success_rate': calc_bucket_rate('medium'),
                'hard_success_rate': calc_bucket_rate('hard'),
                'easy_enabled': difficulty_buckets['easy']['enabled'],
                'medium_enabled': difficulty_buckets['medium']['enabled'],
                'hard_enabled': difficulty_buckets['hard']['enabled'],
            }
        )

        # Adaptive curriculum advancement - check if we should enable harder difficulties
        if i_episode % 100 == 0 and i_episode > 100:
            for idx, (bucket_name, bucket_data) in enumerate(difficulty_buckets.items()):
                if bucket_data['enabled'] and bucket_data['attempts'] >= 50:
                    success_rate = bucket_data['success'] / bucket_data['attempts']

                    # If we've mastered this level, enable the next
                    if success_rate >= curriculum_threshold:
                        # Find next bucket
                        bucket_names = list(difficulty_buckets.keys())
                        if idx < len(bucket_names) - 1:
                            next_bucket = bucket_names[idx + 1]
                            if not difficulty_buckets[next_bucket]['enabled']:
                                difficulty_buckets[next_bucket]['enabled'] = True
                                print(f"\n*** Curriculum Advanced: Enabled '{next_bucket}' difficulty "
                                      f"(mastered '{bucket_name}' with {success_rate*100:.1f}% success) ***")

        # Print progress
        if i_episode % 100 == 0:
            elapsed_time = time.time() - training_start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            enabled_levels = [b for b, info in difficulty_buckets.items() if info['enabled']]
            print(f'Episode {i_episode}/{n_episodes} ({i_episode / n_episodes * 100:.1f}%) | '
                  f'Time: {int(hours)}h {int(minutes)}m {int(seconds)}s | '
                  f'Average Score: {avg_score:.2f} | Epsilon: {eps:.4f} | Enabled: {", ".join(enabled_levels)}')
            print(f'Recent Win Rate: {win_rate:.1f}% | Current Streak: {milestone_tracker.current_win_streak} | Max Streak: {milestone_tracker.max_win_streak}')

        # Save model periodically
        if i_episode % save_every == 0:
            checkpoint_path = f'models/checkpoints/episode_{i_episode}.pth'
            agent.save(checkpoint_path)

            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"Time since last checkpoint: {time_since_last / 60:.1f} minutes")

            # Plot progress
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

            # Plot scores
            ax1.plot(np.arange(len(scores)), scores)
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Episode #')
            ax1.set_title('Training Scores')

            # Plot moving average
            window_size = min(100, len(scores))
            if window_size > 0:
                moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
                ax1.plot(np.arange(window_size - 1, len(scores)), moving_avg, 'r-')

            # Plot behavior success rates
            behavior_names = []
            success_rates = []

            for behavior, stats in behavior_stats.items():
                if stats["attempts"] > 0:
                    behavior_names.append(behavior)
                    success_rates.append(stats["success"] / stats["attempts"] * 100)

            ax2.bar(behavior_names, success_rates)
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_xlabel('Behavior Type')
            ax2.set_title('Success Rate by Fish Behavior')
            ax2.set_ylim(0, 100)

            # Plot loss over time if we have loss data
            if agent.loss_list:
                # Smooth the loss curve with moving average
                window_size = min(100, len(agent.loss_list))
                if window_size > 0:
                    loss_avg = np.convolve(agent.loss_list,
                                           np.ones(window_size) / window_size,
                                           mode='valid')
                    ax3.plot(np.arange(window_size - 1, len(agent.loss_list)), loss_avg)
                    ax3.set_ylabel('Loss')
                    ax3.set_xlabel('Training Steps (x{})'.format(agent.update_every))
                    ax3.set_title('Smoothed Loss Curve')
                    ax3.set_yscale('log')

            plt.tight_layout()
            plt.savefig(f'training_logs/graphs/episode_{i_episode}.png')
            plt.close()

            # Print behavior success stats
            print("\nBehavior success rates:")
            for behavior, stats in behavior_stats.items():
                if stats["attempts"] > 0:
                    print(f"{behavior}: {stats['success']}/{stats['attempts']} "
                          f"({stats['success'] / stats['attempts'] * 100:.1f}%)")

            # Print difficulty bucket success stats
            print("\nDifficulty bucket success rates:")
            for bucket_name, bucket_data in difficulty_buckets.items():
                if bucket_data["attempts"] > 0:
                    success_rate = bucket_data['success'] / bucket_data['attempts'] * 100
                    enabled_str = "✓" if bucket_data['enabled'] else "✗"
                    print(f"{bucket_name} ({enabled_str}): {bucket_data['success']}/{bucket_data['attempts']} "
                          f"({success_rate:.1f}%)")

            # Evaluation for early stopping (only every 1000 episodes for speed)
            if i_episode % 1000 == 0:
                print("\nRunning evaluation for early stopping check...")
                env.render_mode = None  # Ensure no rendering during evaluation
                eval_success_count = 0
                eval_episodes = 20

                for _ in range(eval_episodes):
                    # Select random fish for evaluation
                    fish_name = random.choice(env.get_available_fish())
                    env.fish_name = fish_name
                    state = env.reset()
                    done = False

                    for _ in range(max_t):
                        action = agent.act(state, eps=0.0)  # No exploration during evaluation
                        next_state, reward, done, info = env.step(action)
                        state = next_state

                        if done:
                            if env.distanceFromCatching >= 1.0:  # Success
                                eval_success_count += 1
                            break

                eval_success_rate = eval_success_count / eval_episodes
                print(f"Evaluation success rate: {eval_success_rate * 100:.1f}% ({eval_success_count}/{eval_episodes})")

                # Check for early stopping
                if eval_success_rate >= early_stop_threshold:
                    perfect_episodes += 1
                    print(f"Perfect evaluation round {perfect_episodes}/{required_perfect}")
                    if perfect_episodes >= required_perfect:
                        print(f"\n*** EARLY STOPPING at episode {i_episode} ***")
                        print(
                            f"Achieved {required_perfect} consecutive evaluations with >{early_stop_threshold * 100}% success rate")
                        # Save final model before stopping (checkpoint for reference)
                        agent.save(f'models/checkpoints/early_stop_ep{i_episode}.pth', export_onnx=True)
                        break
                else:
                    perfect_episodes = 0
                    print("Resetting perfect episode counter - continuing training")

            # Restore render mode
            env.render_mode = render_mode_backup

    # Final model saved to checkpoints (manually copy and rename to models/ with descriptive name)
    agent.save(f'models/checkpoints/final_ep{i_episode}.pth', export_onnx=True)
    print(f"\n💾 Final checkpoint saved:")
    print(f"   - PyTorch: models/checkpoints/final_ep{i_episode}.pth")
    print(f"   - ONNX:    models/checkpoints/final_ep{i_episode}.onnx")
    print("   Manually copy both to models/ with a descriptive name (e.g., 'dueling_dqn_95percent_allfish')")

    # Close milestone tracker and save logs
    milestone_tracker.close()

    # Final training stats
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining complete!")
    print(f"Total episodes: {i_episode}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final win rate (last 100 episodes): {win_rate:.1f}%")
    print(f"Max win streak: {milestone_tracker.max_win_streak} episodes")

    return scores

def evaluate_agent(env, agent, n_episodes=20, render=True):
    """Evaluate trained agent performance"""
    scores = []
    success_count = 0
    behavior_results = {}

    # Test on all available fish to get comprehensive performance metrics
    fish_names = env.get_available_fish()

    # Organize by behavior for balanced testing
    behavior_fish = {}
    for fish in env.fish_data:
        behavior = fish["behaviour"]
        if behavior not in behavior_fish:
            behavior_fish[behavior] = []
        behavior_fish[behavior].append(fish["name"])

    # Create test sequence with all behaviors represented
    test_sequence = []
    max_per_behavior = n_episodes // len(behavior_fish) if behavior_fish else 0

    for behavior, fish_list in behavior_fish.items():
        # Add fish from each behavior, up to max_per_behavior
        for i in range(min(max_per_behavior, len(fish_list))):
            test_sequence.append(fish_list[i])

    # Fill remaining slots with random fish
    while len(test_sequence) < n_episodes:
        test_sequence.append(random.choice(fish_names))

    # Randomize order
    random.shuffle(test_sequence)

    for i_episode, fish_name in enumerate(test_sequence):
        env.fish_name = fish_name
        state = env.reset()

        behavior = env.current_fish["behaviour"]
        difficulty = env.current_fish["difficulty"]

        if behavior not in behavior_results:
            behavior_results[behavior] = {"attempts": 0, "success": 0}
        behavior_results[behavior]["attempts"] += 1

        print(f"\nEvaluation {i_episode + 1}/{n_episodes}: {fish_name} "
              f"(Behavior: {behavior}, Difficulty: {difficulty})")

        score = 0

        for t in range(1000):  # max steps per episode
            action = agent.act(state, eps=0.0)  # no exploration in evaluation
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward

            if render and env.render_mode == "human":
                time.sleep(0.01)  # slow down rendering

            if done:
                if env.distanceFromCatching >= 1.0:
                    print(f"Success! Score: {score:.2f}")
                    success_count += 1
                    behavior_results[behavior]["success"] += 1
                else:
                    print(f"Failed. Score: {score:.2f}")
                break

        scores.append(score)

    print(f"\nEvaluation complete. Overall success rate: {success_count}/{n_episodes} "
          f"({success_count / n_episodes * 100:.1f}%)")
    print(f"Average score: {np.mean(scores):.2f}")

    print("\nPerformance by behavior type:")
    for behavior, results in behavior_results.items():
        success_rate = results["success"] / results["attempts"] * 100 if results["attempts"] > 0 else 0
        print(f"{behavior}: {results['success']}/{results['attempts']} ({success_rate:.1f}%)")

    return scores, behavior_results


if __name__ == "__main__":
    # Create environment and agent with optimized parameters
    # For vectorized training (2-4x speedup), uncomment below and modify training loop:
    # env = VectorizedEnv(num_envs=4)
    env = FishingMinigameEnv(render_mode="human")

    # Create agent with all improvements: Dueling DQN, Prioritized Replay, N-step returns
    # State dimension now 14: added 3 temporal features (bobber accel, bar accel, distance to bar)
    agent = DQNAgent(
        state_dim=14,  # Updated from 11 to 14 for temporal features
        action_dim=2,
        hidden_sizes=[128, 128, 64],
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=100000,
        batch_size=128,  # Increased from 128 for speed (6GB VRAM)
        update_every=4,
        n_step=3,  # 3-step returns
        target_update_freq=1000  # Hard update every 1000 steps
    )

    # Training modes
    train_new_model = True  # Set to True to train from scratch
    fine_tune_model = False  # Set to True to continue training with new fish

    if train_new_model:
        scores = train_dqn(
            env=env,
            agent=agent,
            n_episodes=10000,           # Large enough for overnight training
            max_t=2000,                 # Maximum timesteps per episode
            eps_start=0.2,              # Start with good exploration (cosine annealing)
            eps_end=0.001,              # Lower final exploration
            save_every=500,             # Save checkpoints (optimized for speed)
            render_every=2000           # Occasional visual check (optimized for speed)
        )
        # Training complete - manually name your model in models/ folder

    elif fine_tune_model:
        # Load pre-trained model and continue training (for new fish)
        # TODO: Change to your model name in models/ folder
        model_path = 'models/YOUR_MODEL_NAME.pth'  # ← CHANGE THIS
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded model from {model_path}")
            print("Starting fine-tuning with reduced learning rate and augmentation...")

            # Reduce learning rate for fine-tuning
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = 1e-4  # Lower learning rate

            # Continue training
            scores = train_dqn(
                env=env,
                agent=agent,
                n_episodes=2000,            # Fewer episodes for fine-tuning
                max_t=2000,
                eps_start=0.05,             # Lower exploration (already knows basics)
                eps_end=0.001,
                save_every=500,
                render_every=1000
            )
            # Training complete - manually name your finetuned model in models/ folder
        else:
            print(f"Model file {model_path} not found. Cannot fine-tune.")

    else:
        # Load pre-trained model for evaluation only
        # TODO: Change to your model name in models/ folder
        model_path = 'models/YOUR_MODEL_NAME.pth'  # ← CHANGE THIS
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found.")
            print("Available models in models/ folder:")
            if os.path.exists('models'):
                models = [f for f in os.listdir('models') if f.endswith('.pth')]
                for m in models:
                    print(f"  - {m}")
            exit(1)

    # Note: ONNX export is now automatic when saving models
    # Both .pth and .onnx files are created with the same base name

    # Evaluate agent performance
    print("\nRunning comprehensive evaluation...")
    scores, behavior_results = evaluate_agent(env, agent, n_episodes=20, render=True)

    # Interactive mode
    print("\nEntering interactive mode. Select a fish to watch the agent catch, or 'q' to quit.")

    while True:
        # Display fish options with behavior and difficulty
        print("\nAvailable fish:")
        for i, fish_name in enumerate(env.get_available_fish()):
            # Find fish details
            fish_detail = next((f for f in env.fish_data if f["name"] == fish_name), None)
            if fish_detail:
                behavior = fish_detail.get("behaviour", "mixed")
                difficulty = fish_detail.get("difficulty", 0)
                print(f"{i}: {fish_name} (Behavior: {behavior}, Difficulty: {difficulty})")
            else:
                print(f"{i}: {fish_name}")

        # Get user selection
        key = input("\nSelect fish number, 'r' for random, or 'q' to quit: ")

        if key.lower() == 'q':
            break

        if key.lower() == 'r':
            fish_name = random.choice(env.get_available_fish())
        elif key.isdigit() and 0 <= int(key) < len(env.get_available_fish()):
            fish_name = env.get_available_fish()[int(key)]
        else:
            print("Invalid selection.")
            continue

        # Set fish and reset
        env.fish_name = fish_name
        state = env.reset()

        # Get fish details for display
        behavior = env.current_fish["behaviour"]
        difficulty = env.current_fish["difficulty"]

        print(f"\nWatching agent catch: {fish_name} (Behavior: {behavior}, Difficulty: {difficulty})")

        score = 0
        done = False

        # Agent plays until done
        while not done:
            action = agent.act(state, eps=0.0)  # no exploration in evaluation
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward

            time.sleep(0.016)  # ~60 FPS for smooth playback

        # Display result
        result = "Success!" if env.distanceFromCatching >= 1.0 else "Failed"
        print(f"{result} Score: {score:.2f}")
        time.sleep(1)  # Pause to see result

    # Clean up
    env.close()
