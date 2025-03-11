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


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        experiences = random.sample(self.buffer, k=batch_size)

        # Convert to tensors for batch processing
        states = torch.tensor(np.vstack([e.state for e in experiences]), dtype=torch.float32).to(device)
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.long).to(device)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences]), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.float32).to(device)
        dones = torch.tensor(np.vstack([e.done for e in experiences]).astype(np.uint8), dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)


class DQN(nn.Module):
    """Enhanced Deep Q-Network model"""

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=[128, 128, 64]):
        super(DQN, self).__init__()

        # Enhanced neural network architecture
        layers = []
        input_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for ReLU networks
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class DQNAgent:
    """Agent implementing Double DQN algorithm with optimizations"""

    def __init__(self, state_dim=10, action_dim=2, hidden_sizes=[128, 128, 64], learning_rate=3e-4, gamma=0.99,
                 tau=5e-3, buffer_size=100000, batch_size=128, update_every=4):
        """Initialize agent parameters and build models"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update parameter

        # Q-Networks with enhanced architecture
        self.qnetwork_local = DQN(state_dim, action_dim, hidden_sizes).to(device)
        self.qnetwork_target = DQN(state_dim, action_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Larger replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = update_every

        # For tracking statistics
        self.loss_list = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample to learn"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                loss = self.learn(experiences)
                self.loss_list.append(loss)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy

        Args:
            state: current state
            eps: epsilon for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def learn(self, experiences):
        """Update value parameters using batch of experience tuples with Double DQN

        Args:
            experiences: tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: use local network to select actions, target network to evaluate
        with torch.no_grad():
            # Get action indices from local model (best actions)
            action_indices = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)

            # Get Q values from target model for those actions
            Q_targets_next = self.qnetwork_target(next_states).gather(1, action_indices)

            # Compute Q targets for current states using Bellman equation
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)

        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # Return loss value for monitoring
        return loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """Save trained model"""
        torch.save({'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list}, filename)

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


def train_dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.998, save_every=100,
              render_every=100):
    """Train DQN agent with curriculum learning

    Args:
        env: environment
        agent: DQN agent
        n_episodes: maximum number of training episodes
        max_t: maximum number of timesteps per episode
        eps_start: starting value of epsilon for epsilon-greedy action selection
        eps_end: minimum value of epsilon
        eps_decay: multiplicative factor for decreasing epsilon
        save_every: how often to save the model (episodes)
        render_every: how often to render an episode
    """
    scores = []  # list of scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for tracking progress
    eps = eps_start

    # Create directory for saving models
    os.makedirs("models", exist_ok=True)

    # Track difficulties mastered
    difficulty_success = {}

    # For plotting
    fish_behaviors = list(env.BEHAVIOR_TYPES.keys())
    behavior_stats = {b: {"attempts": 0, "success": 0} for b in fish_behaviors}

    # For curriculum learning
    phase = 1
    phase_thresholds = {1: {'score': 5.0, 'episodes': 500, 'difficulty_max': 40},
        2: {'score': 7.0, 'episodes': 1000, 'difficulty_max': 70},
        3: {'score': 10.0, 'episodes': 1500, 'difficulty_max': 100}}

    for i_episode in range(1, n_episodes + 1):
        # Curriculum learning - select appropriate fish based on current phase
        available_fish = [f for f in env.fish_data if f["difficulty"] <= phase_thresholds[phase]['difficulty_max']]

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
                env.root.update()
                time.sleep(0.01)  # slow down rendering

            if done:
                if env.distanceFromCatching >= 1.0:  # Successfully caught fish
                    behavior_stats[fish_behavior]["success"] += 1

                    # Track difficulty mastery
                    difficulty_str = f"{fish_difficulty}"
                    if difficulty_str not in difficulty_success:
                        difficulty_success[difficulty_str] = {"attempts": 0, "success": 0}
                    difficulty_success[difficulty_str]["attempts"] += 1
                    difficulty_success[difficulty_str]["success"] += 1
                else:
                    # Track failed attempt
                    difficulty_str = f"{fish_difficulty}"
                    if difficulty_str not in difficulty_success:
                        difficulty_success[difficulty_str] = {"attempts": 0, "success": 0}
                    difficulty_success[difficulty_str]["attempts"] += 1

                break

        # Restore render mode
        env.render_mode = render_mode_backup

        # Record score and update epsilon
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        # Check if we should advance curriculum phase
        avg_score = np.mean(scores_window)
        current_episode = i_episode

        if (phase < 3 and avg_score >= phase_thresholds[phase]['score'] and current_episode >= phase_thresholds[phase][
            'episodes']):
            phase += 1
            print(f"Advancing to phase {phase} - introducing more difficult fish!")

        # Print progress
        if i_episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.4f}\tPhase: {phase}')

            # Calculate success rate over last 100 episodes
            success_count = sum(1 for i in range(max(0, len(scores) - 100), len(scores)) if scores[i] > 0)
            win_rate = success_count / min(100, len(scores)) * 100
            print(f'Recent Win Rate: {win_rate:.1f}%')

        # Save model periodically
        if i_episode % save_every == 0:
            agent.save(f'models/dqn_fishing_episode_{i_episode}.pth')

            # Plot progress
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

            # Plot scores
            ax1.plot(np.arange(len(scores)), scores)
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Episode #')
            ax1.set_title('Training Scores')

            # Plot moving average
            window_size = min(100, len(scores))
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
                    loss_avg = np.convolve(agent.loss_list, np.ones(window_size) / window_size, mode='valid')
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

            # Print difficulty success stats
            print("\nDifficulty success rates:")
            for diff, stats in sorted(difficulty_success.items()):
                if stats["attempts"] > 0:
                    print(f"Difficulty {diff}: {stats['success']}/{stats['attempts']} "
                          f"({stats['success'] / stats['attempts'] * 100:.1f}%)")

    # Save final model
    agent.save('models/dqn_fishing_final.pth')
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

    agent = DQNAgent(state_dim=10,  # Match observation space
        action_dim=2,  # Press/no press
        hidden_sizes=[128, 128, 64],  # Larger network architecture
        learning_rate=3e-4,  # Optimized learning rate
        gamma=0.99,  # High discount for long-term rewards
        tau=5e-3,  # Faster target updates
        buffer_size=100000,  # Large experience buffer
        batch_size=128,  # Larger batches for stable learning
        update_every=4  # Learn every 4 steps
    )

    # Train or load model
    train_new_model = True  # Set to False to load a saved model

    if train_new_model:
        scores = train_dqn(env=env, agent=agent, n_episodes=5000,  # More episodes for better learning
            max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.998,  # Slower decay for better exploration
            save_every=100, render_every=200)
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
                behavior = fish_detail.get("behaviour", "unknown")
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

            env.root.update()
            time.sleep(0.016)  # ~60 FPS for smooth playback

        # Display result
        result = "Success!" if env.distanceFromCatching >= 1.0 else "Failed"
        print(f"{result} Score: {score:.2f}")
        time.sleep(1)  # Pause to see result

    # Clean up
    env.close()
