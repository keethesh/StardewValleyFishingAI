import os
import random
import time
from collections import deque, namedtuple

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
        self.tree[parent] += change
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
    """Dueling Deep Q-Network with Layer Normalization"""

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=[128, 128, 64]):
        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim

        # Shared feature extraction layers with Layer Normalization
        feature_layers = []
        input_size = state_dim

        for hidden_size in hidden_sizes:
            feature_layers.append(nn.Linear(input_size, hidden_size))
            feature_layers.append(nn.LayerNorm(hidden_size))
            feature_layers.append(nn.ReLU())
            input_size = hidden_size

        self.feature_layer = nn.Sequential(*feature_layers)

        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
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

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=[128, 128, 64], learning_rate=3e-4, gamma=0.99,
                 buffer_size=100000, batch_size=128, update_every=4, n_step=3, target_update_freq=1000):
        """Initialize agent parameters and build models"""
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

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-5)

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

        # Get expected Q values from local model
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

    def save(self, filename):
        """Save trained model"""
        torch.save({'local_state_dict': self.qnetwork_local.state_dict(),
                    'target_state_dict': self.qnetwork_target.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss_list': self.loss_list}, filename)

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

    # Create directory for saving models
    os.makedirs("models", exist_ok=True)

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

            if render and env.root is not None:
                env.root.update_idletasks()  # Optimized rendering
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
            avg_score = np.mean(scores_window)
            elapsed_time = time.time() - training_start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            enabled_levels = [b for b, info in difficulty_buckets.items() if info['enabled']]
            print(f'Episode {i_episode}/{n_episodes} ({i_episode / n_episodes * 100:.1f}%) | '
                  f'Time: {int(hours)}h {int(minutes)}m {int(seconds)}s | '
                  f'Average Score: {avg_score:.2f} | Epsilon: {eps:.4f} | Enabled: {", ".join(enabled_levels)}')

            # Calculate success rate over last 100 episodes
            success_count = sum(1 for i in range(max(0, len(scores) - 100), len(scores))
                                if scores[i] > 0)
            win_rate = success_count / min(100, len(scores)) * 100
            print(f'Recent Win Rate: {win_rate:.1f}%')

        # Save model periodically
        if i_episode % save_every == 0:
            checkpoint_path = f'models/dqn_fishing_episode_{i_episode}.pth'
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
            plt.savefig(f'models/training_progress_{i_episode}.png')
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

            # Evaluation for early stopping
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
                    # Save final model before stopping
                    agent.save('models/dqn_fishing_final.pth')
                    break
            else:
                perfect_episodes = 0
                print("Resetting perfect episode counter - continuing training")

            # Restore render mode
            env.render_mode = render_mode_backup

    # Save final model
    agent.save('models/dqn_fishing_final.pth')

    # Final training stats
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining complete!")
    print(f"Total episodes: {i_episode}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final win rate (last 100 episodes): {win_rate:.1f}%")

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
                env.root.update()
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
        batch_size=128,
        update_every=4,
        n_step=3,  # 3-step returns
        target_update_freq=1000  # Hard update every 1000 steps
    )

    # Train or load model
    train_new_model = False  # Set to False to load a saved model

    if train_new_model:
        scores = train_dqn(
            env=env,
            agent=agent,
            n_episodes=10000,           # Large enough for overnight training
            max_t=2000,                 # Maximum timesteps per episode
            eps_start=0.2,              # Start with good exploration (cosine annealing)
            eps_end=0.001,              # Lower final exploration
            save_every=500,             # Save checkpoints regularly
            render_every=1000           # Occasional visual check
        )
        agent.save('models/dqn_fishing_final.pth')
    else:
        # Load pre-trained model
        model_path = 'models/dqn_fishing_final.pth'  # Change to desired model file
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found. Training new model instead.")
            train_new_model = True

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

            if env.root is not None:
                env.root.update_idletasks()  # Optimized rendering
            time.sleep(0.016)  # ~60 FPS for smooth playback

        # Display result
        result = "Success!" if env.distanceFromCatching >= 1.0 else "Failed"
        print(f"{result} Score: {score:.2f}")
        time.sleep(1)  # Pause to see result

    # Clean up
    env.close()
