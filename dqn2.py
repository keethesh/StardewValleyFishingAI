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
from tqdm import tqdm

from environment import FishingMinigameEnv

# Assume FishingMinigameEnv is available from previous code

# Set up device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the experience tuple structure
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""

    def __init__(self, capacity=10000):
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
    """Deep Q-Network model"""

    def __init__(self, state_dim=10, action_dim=2, hidden_size=64):
        super(DQN, self).__init__()

        # Neural network architecture
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, action_dim))

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class DQNAgent:
    """Agent implementing DQN algorithm"""

    def __init__(self, state_dim=10, action_dim=2, hidden_size=64, learning_rate=1e-3, gamma=0.99, tau=1e-3,
                 buffer_size=10000, batch_size=64):
        """Initialize agent parameters and build models"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update parameter

        # Q-Networks
        self.qnetwork_local = DQN(state_dim, action_dim, hidden_size).to(device)
        self.qnetwork_target = DQN(state_dim, action_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = 4

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
                self.learn(experiences)

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
        """Update value parameters using batch of experience tuples

        Args:
            experiences: tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

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
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        """Load trained model"""
        self.qnetwork_local.load_state_dict(torch.load(filename))
        self.qnetwork_target.load_state_dict(torch.load(filename))


def train_dqn(env, agent, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_every=100,
              render_every=100):
    """Train DQN agent

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

    for i_episode in range(1, n_episodes + 1):
        # Reset environment with random fish
        fish_name = random.choice(env.get_available_fish())
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
            print(f"\nEpisode {i_episode}/{n_episodes}: Rendering... Fish: {fish_name} ({fish_behavior})")
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

        # Print progress
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}')

        # Save model periodically
        if i_episode % save_every == 0:
            agent.save(f'models/dqn_fishing_episode_{i_episode}.pth')

            # Plot progress
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot scores
            ax1.plot(np.arange(len(scores)), scores)
            ax1.set_ylabel('Score')
            ax1.set_xlabel('Episode #')
            ax1.set_title('Training Scores')

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

    return scores


def evaluate_agent(env, agent, n_episodes=10, render=True):
    """Evaluate trained agent performance"""
    scores = []
    success_count = 0

    # Get one of each behavior type for testing
    behavior_fish = {}
    for fish in env.fish_data:
        behavior = fish["behaviour"]
        if behavior not in behavior_fish:
            behavior_fish[behavior] = fish["name"]

    fish_names = list(behavior_fish.values())
    if not fish_names:  # Fallback if behaviors aren't found
        fish_names = env.get_available_fish()

    for i_episode in range(n_episodes):
        # Rotate through fish types
        fish_name = fish_names[i_episode % len(fish_names)]
        env.fish_name = fish_name
        state = env.reset()

        print(f"\nEvaluating on: {fish_name} ({env.current_fish['behaviour']})")
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
                else:
                    print(f"Failed. Score: {score:.2f}")
                break

        scores.append(score)

    print(f"\nEvaluation complete. Success rate: {success_count}/{n_episodes} "
          f"({success_count / n_episodes * 100:.1f}%)")
    print(f"Average score: {np.mean(scores):.2f}")

    return scores


if __name__ == "__main__":
    # Create environment and agent
    env = FishingMinigameEnv(render_mode="human")
    agent = DQNAgent(state_dim=10,  # Match observation space
        action_dim=2,  # Press/no press
        hidden_size=64,  # Size of hidden layers
        learning_rate=5e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        tau=1e-3,  # Soft update parameter
        buffer_size=50000,  # Replay buffer size
        batch_size=64  # Minibatch size
    )

    # Train or load model
    train_new_model = True  # Set to False to load a saved model

    if train_new_model:
        scores = train_dqn(env=env, agent=agent, n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01,
            eps_decay=0.995, save_every=100, render_every=200)
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
    if not train_new_model:  # Only evaluate if we're using a pre-trained model
        evaluate_agent(env, agent, n_episodes=5, render=True)

    # Add an interactive mode to watch the agent play
    print("\nEntering interactive mode. Press q to quit, or press any key to watch agent catch a fish.")

    while True:
        key = input("Select fish (enter number or 'r' for random, 'q' to quit): ")

        if key.lower() == 'q':
            break

        if key.lower() == 'r':
            fish_name = random.choice(env.get_available_fish())
        elif key.isdigit() and 0 <= int(key) < len(env.get_available_fish()):
            fish_name = env.get_available_fish()[int(key)]
        else:
            print("Available fish:")
            for i, fish in enumerate(env.get_available_fish()):
                print(f"{i}: {fish}")
            continue

        env.fish_name = fish_name
        state = env.reset()

        print(f"Watching agent catch: {fish_name} ({env.current_fish['behaviour']})")
        score = 0

        for t in range(1000):  # max steps per episode
            action = agent.act(state, eps=0.0)  # no exploration in evaluation
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward

            env.root.update()
            time.sleep(0.016)  # ~60 FPS for smooth playback

            if done:
                result = "Success!" if env.distanceFromCatching >= 1.0 else "Failed"
                print(f"{result} Score: {score:.2f}")
                time.sleep(1)  # Pause to see result
                break

    # Clean up
    env.close()
