import json
import os
import time
import tkinter as tk

import numpy as np


class FishingMinigameEnv:
    """
    A fishing minigame environment designed for reinforcement learning.
    Follows a similar API to OpenAI Gym/Gymnasium.
    """

    # Action space constants
    ACTION_NONE = 0
    ACTION_PRESS = 1

    # Behavior type mapping
    BEHAVIOR_TYPES = {"mixed": 0,  # Default mixed behavior
        "dart": 1,  # Darting movement with sudden direction changes
        "smooth": 2,  # Mostly static with occasional movement
        "sinker": 3,  # Tends to sink
        "floater": 4  # Tends to float
    }

    def __init__(self, render_mode="human", seed=None, fish_name=None):
        # Set random seed if provided
        self.np_random = np.random.RandomState(seed)

        # Environment parameters
        self.track_height = 568
        self.track_width = 100

        # Load fish data
        self.fish_data = self.load_fish_data()

        # Current fish info
        self.current_fish = None
        self.fish_name = fish_name  # Will select a specific fish if provided

        # Rendering setup
        self.render_mode = render_mode
        self.root = None
        self.canvas = None
        self.button_pressed = False  # Track button state for human play

        # For tracking ML training progress
        self.episode_reward = 0
        self.episode_length = 0

        # Create GUI if needed
        if render_mode == "human":
            self.setup_gui()

        # Initialize the environment
        self.reset()

    def load_fish_data(self):
        """Load fish data from fish.json file."""
        try:
            with open("fish.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading fish.json: {e}")
            # Provide default fish data if file not found or invalid
            return [{"name": "Default Fish", "difficulty": 50, "behaviour": "mixed"}]

    def select_fish(self, fish_name=None):
        """Select a fish by name or randomly."""
        if fish_name:
            # Find fish by name
            for fish in self.fish_data:
                if fish["name"].lower() == fish_name.lower():
                    return fish

        # Select random fish if not found or none specified
        if self.fish_data:
            return self.np_random.choice(self.fish_data)
        else:
            return {"name": "Default Fish", "difficulty": 50, "behaviour": "mixed"}

    def get_motion_type(self, behaviour):
        """Convert behaviour string to motion type."""
        behaviour = behaviour.lower()
        return self.BEHAVIOR_TYPES.get(behaviour, 0)  # Default to mixed if unknown

    def setup_gui(self):
        """Setup the GUI components."""
        self.root = tk.Tk()
        self.root.title("Fishing Minigame (ML Environment)")
        self.canvas = tk.Canvas(self.root, width=self.track_width, height=self.track_height, bg="lightblue")
        self.canvas.pack()

        # Setup key bindings for human play
        self.root.bind("<KeyPress-space>", self.on_key_press)
        self.root.bind("<KeyRelease-space>", self.on_key_release)
        self.root.bind("<ButtonPress-1>", self.on_mouse_press)
        self.root.bind("<ButtonRelease-1>", self.on_mouse_release)

    def on_key_press(self, event):
        """Handle key press events for human play"""
        self.button_pressed = True

    def on_key_release(self, event):
        """Handle key release events for human play"""
        self.button_pressed = False

    def on_mouse_press(self, event):
        """Handle mouse press events for human play"""
        self.button_pressed = True

    def on_mouse_release(self, event):
        """Handle mouse release events for human play"""
        self.button_pressed = False

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        """Reset the environment to initial state and return observation."""
        self.current_timestep = 0
        self.max_timesteps = 2000  # Match with max_t

        # Select a fish
        self.current_fish = self.select_fish(self.fish_name)

        # Game logic variables
        self.difficulty = self.current_fish["difficulty"]
        self.motionType = self.get_motion_type(self.current_fish["behaviour"])
        self.whichBobber = 0
        self.floaterSinkerAcceleration = 0.0
        self.bobberSpeed = 0.0
        self.bobberAcceleration = 0.0
        self.bobberPosition = 100.0
        self.bobberTargetPosition = 200.0
        self.bobberBarPos = 200.0
        self.bobberBarSpeed = 0.0
        self.bobberBarHeight = 96
        self.minFishSize = 5
        self.maxFishSize = 20
        self.fishSize = 10
        self.fishSizeReductionTimer = 800
        self.beginnersRod = False
        self.perfect = True
        self.treasure = False
        self.treasureCaught = False
        self.treasureScale = 0.0
        self.treasureAppearTimer = 2000
        self.treasurePosition = 0.0
        self.treasureCatchLevel = 0.0
        self.distanceFromCatching = 0.5
        self.handledFishResult = False
        self.bobberInBar = False
        self.done = False

        # Reset episode tracking
        self.episode_reward = 0
        self.episode_length = 0

        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        """Convert game state to ML-friendly observation vector."""
        # Add normalized time step to observation
        obs = np.array([self.bobberPosition / self.track_height,  # normalized fish position
                        self.bobberSpeed / 10.0,  # normalized fish speed
                        self.bobberBarPos / self.track_height,  # normalized bar position
                        self.bobberBarSpeed / 10.0,  # normalized bar speed
                        self.bobberBarHeight / self.track_height,  # normalized bar height
            float(self.bobberInBar),  # binary: fish in bar?
            self.distanceFromCatching,  # progress toward catching (0-1)
                        self.fishSize / self.maxFishSize,  # normalized fish size
                        self.difficulty / 100.0,  # normalized difficulty
                        float(self.motionType) / 4.0,  # normalized motion type
                        self.current_timestep / self.max_timesteps,  # normalized time progress
        ], dtype=np.float32)

        return obs

    def step(self, action):
        # Save previous state for reward calculation
        prev_distance = self.distanceFromCatching

        # Apply action
        button_pressed = (action == self.ACTION_PRESS)

        # Update the environment
        self._update_game_logic(16, button_pressed)  # 16ms timestep (~60 FPS)

        # Increment timestep counter
        self.current_timestep += 1

        # Calculate reward
        reward = self._calculate_reward(prev_distance)
        self.episode_reward += reward
        self.episode_length += 1

        # Check for episode termination
        done = self.handledFishResult or (self.current_timestep >= self.max_timesteps)
        self.done = done

        # If timeout, set handledFishResult to true to indicate failure
        if self.current_timestep >= self.max_timesteps and not self.handledFishResult:
            self.handledFishResult = True
            self.distanceFromCatching = 0.0  # Ensure it's treated as a failure

        # Get new observation
        obs = self._get_observation()

        # Additional info for debugging and analysis
        info = {"fish_name": self.current_fish["name"], "fish_difficulty": self.difficulty,
                "fish_behaviour": self.current_fish["behaviour"], "fish_size": self.fishSize,
                "distance_from_catching": self.distanceFromCatching, "bobber_in_bar": self.bobberInBar,
                "episode_length": self.episode_length, "episode_reward": self.episode_reward, }

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            self.root.update()

        return obs, reward, done, info

    def _calculate_reward(self, prev_distance):
        """Calculate reward based on current state and previous state."""
        # Progress reward: improvement in catching progress
        progress_reward = (self.distanceFromCatching - prev_distance) * 10.0

        # Reward for keeping fish in bar
        in_bar_reward = 0.1 if self.bobberInBar else -0.05

        # Penalty for extreme movements
        movement_penalty = -0.01 * abs(self.bobberBarSpeed)

        # NEW: Time efficiency penalty - encourages faster catches
        time_penalty = -0.01  # Small constant penalty per timestep

        # Scale rewards based on difficulty
        difficulty_factor = self.difficulty / 50.0  # Higher difficulty = higher rewards

        # Terminal rewards
        if self.handledFishResult:
            if self.distanceFromCatching >= 1.0:  # Success
                # Bonus for faster catches (scaled by remaining time)
                time_bonus = 5.0 * (1.0 - (self.current_timestep / self.max_timesteps))
                return (10.0 * difficulty_factor + (self.fishSize / self.maxFishSize) * 10.0 + time_bonus)
            else:  # Failure
                return -5.0

        return (progress_reward + in_bar_reward + movement_penalty + time_penalty) * difficulty_factor

    def _update_game_logic(self, time_elapsed, button_pressed):
        """Update game state based on elapsed time and inputs."""
        # Attempt to set a new target occasionally
        if (self.np_random.random() < (self.difficulty * (20.0 if self.motionType == 2 else 1.0)) / 4000.0 and (
                self.motionType != 2 or self.bobberTargetPosition == -1.0)):
            num1 = 548.0 - self.bobberPosition
            bobberPos = self.bobberPosition
            num2 = min(99.0, self.difficulty + self.np_random.randint(10, 45)) / 100.0
            self.bobberTargetPosition = self.bobberPosition + self.np_random.randint(int(max(-bobberPos, -num1)),
                                                                                     int(num1)) * num2

        # Floater/sinker adjustments
        if self.motionType == 4:  # Floater
            self.floaterSinkerAcceleration = max(self.floaterSinkerAcceleration - 0.01, -1.5)
        elif self.motionType == 3:  # Sinker
            self.floaterSinkerAcceleration = min(self.floaterSinkerAcceleration + 0.01, 1.5)

        # Move bobber towards target
        if abs(self.bobberPosition - self.bobberTargetPosition) > 3.0 and self.bobberTargetPosition != -1.0:
            self.bobberAcceleration = ((self.bobberTargetPosition - self.bobberPosition) / (
                    self.np_random.randint(10, 30) + (100.0 - min(100.0, self.difficulty))))
            self.bobberSpeed += (self.bobberAcceleration - self.bobberSpeed) / 5.0
        else:
            # If no target, set a random one based on difficulty
            if self.motionType == 2 or self.np_random.random() >= self.difficulty / 2000.0:
                self.bobberTargetPosition = -1.0
            else:
                self.bobberTargetPosition = self.bobberPosition + (
                    self.np_random.randint(-100, -51) if self.np_random.random() < 0.5 else self.np_random.randint(50,
                                                                                                                   101))

        if self.motionType == 1 and self.np_random.random() < self.difficulty / 1000.0:
            self.bobberTargetPosition = self.bobberPosition + (self.np_random.randint(-100 - int(self.difficulty) * 2,
                                                                                      -51) if self.np_random.random() < 0.5 else self.np_random.randint(
                50, 101 + int(self.difficulty) * 2))

        # Clamp bobber target
        self.bobberTargetPosition = max(-1.0, min(self.bobberTargetPosition, 548.0))

        # Update bobber position
        self.bobberPosition += self.bobberSpeed + self.floaterSinkerAcceleration
        self.bobberPosition = max(0.0, min(self.bobberPosition, 532.0))

        # Check if bobber in bar - FIXED for better centering
        fish_center = self.bobberPosition
        bar_top = self.bobberBarPos
        bar_bottom = self.bobberBarPos + self.bobberBarHeight

        # Fish is in bar if its center is within the bar's range
        self.bobberInBar = (fish_center >= bar_top and fish_center <= bar_bottom)

        # Move the bobber bar based on input
        num4 = -0.25 if button_pressed else 0.25
        if button_pressed and num4 < 0.0 and (
                self.bobberBarPos == 0.0 or self.bobberBarPos == (568 - self.bobberBarHeight)):
            self.bobberBarSpeed = 0.0

        if self.bobberInBar:
            num4 *= (0.3 if self.whichBobber == 691 else 0.6)
            if self.whichBobber == 691:
                mid_point = self.bobberBarPos + (self.bobberBarHeight / 2.0)
                if self.bobberPosition < mid_point:
                    self.bobberBarSpeed -= 0.2
                else:
                    self.bobberBarSpeed += 0.2

        self.bobberBarSpeed += num4
        self.bobberBarPos += self.bobberBarSpeed

        # Constrain the bar
        if self.bobberBarPos + self.bobberBarHeight > 568.0:
            self.bobberBarPos = 568.0 - self.bobberBarHeight
            self.bobberBarSpeed = -(self.bobberBarSpeed * 2.0 / 3.0 * (0.1 if self.whichBobber == 692 else 1.0))
        elif self.bobberBarPos < 0.0:
            self.bobberBarPos = 0.0
            self.bobberBarSpeed = -(self.bobberBarSpeed * 2.0 / 3.0)

        # Update distance from catching
        if self.bobberInBar:
            self.distanceFromCatching += (1.0 / 500.0)
        else:
            self.fishSizeReductionTimer -= time_elapsed
            if self.fishSizeReductionTimer <= 0:
                self.fishSize = max(self.minFishSize, self.fishSize - 1)
                self.fishSizeReductionTimer = 800
            self.distanceFromCatching -= (
                1.0 / 500.0 if (self.whichBobber == 694 or self.beginnersRod) else 3.0 / 1000.0)

        self.distanceFromCatching = max(0.0, min(1.0, self.distanceFromCatching))

        # Check win/lose
        if self.distanceFromCatching <= 0.0 or self.distanceFromCatching >= 1.0:
            self.handledFishResult = True

    def _render_frame(self):
        """Render the current frame to the canvas."""
        if self.canvas is None:
            return

        self.canvas.delete("all")

        # Draw track
        self.canvas.create_rectangle(0, 0, self.track_width, self.track_height, fill="white", outline="black")

        # Draw bobber bar
        bar_x1 = 20
        bar_x2 = 80
        bar_y1 = self.bobberBarPos
        bar_y2 = self.bobberBarPos + self.bobberBarHeight
        self.canvas.create_rectangle(bar_x1, bar_y1, bar_x2, bar_y2, fill="green" if self.bobberInBar else "orange",
                                     outline="black")

        # Draw fish (centered at middle of track)
        fish_x = self.track_width // 2
        fish_radius = 10
        fish_y = self.bobberPosition
        self.canvas.create_oval(fish_x - fish_radius, fish_y - fish_radius, fish_x + fish_radius, fish_y + fish_radius,
                                fill="red", outline="black")

        # Draw progress bar
        progress_width = 10
        progress_x = 85
        progress_height = int(self.track_height * self.distanceFromCatching)
        progress_y = self.track_height - progress_height
        self.canvas.create_rectangle(progress_x, progress_y, progress_x + progress_width, self.track_height,
                                     fill="blue", outline="black")

        # Display fish info
        self.canvas.create_text(50, 10, text=f"Fish: {self.current_fish['name']}", fill="black")
        self.canvas.create_text(50, 30, text=f"Difficulty: {self.difficulty}", fill="black")
        self.canvas.create_text(50, 50, text=f"Progress: {self.distanceFromCatching:.2f}", fill="black")

        # Display controls help
        if self.done:
            result = "Success!" if self.distanceFromCatching >= 1.0 else "Failed!"
            self.canvas.create_text(50, 70, text=f"Game Over - {result}", fill="red")
            self.canvas.create_text(50, 90, text="Press R to restart", fill="red")
        else:
            self.canvas.create_text(50, 70, text="Hold SPACE to pull", fill="black")

    def close(self):
        """Close the environment and clean up resources."""
        if self.root is not None:
            self.root.destroy()
            self.root = None
            self.canvas = None

    def get_available_fish(self):
        """Return list of available fish names."""
        return [fish["name"] for fish in self.fish_data]


# Create a default fish.json file if it doesn't exist
def create_default_fish_file():
    """Create a default fish.json file if it doesn't exist."""
    if not os.path.exists("fish.json"):
        default_fish = {"fish": [{"name": "Pufferfish", "difficulty": 80, "behaviour": "floater"},
                                 {"name": "Salmon", "difficulty": 50, "behaviour": "mixed"},
                                 {"name": "Octopus", "difficulty": 95, "behaviour": "sinker"},
                                 {"name": "Trout", "difficulty": 30, "behaviour": "mixed"},
                                 {"name": "Shark", "difficulty": 90, "behaviour": "dart"}]}

        try:
            with open("fish.json", "w") as f:
                json.dump(default_fish, f, indent=2)
            print("Created default fish.json file")
        except Exception as e:
            print(f"Error creating fish.json: {e}")


# Example of how to collect training data with various fish
def collect_training_data(episodes=100, max_steps=1000, render=True):
    """Collect state-action-reward data for ML training."""
    env = FishingMinigameEnv(render_mode="human" if render else None)
    all_data = []

    # Get list of available fish
    available_fish = env.get_available_fish()

    for episode in range(episodes):
        # Select a random fish for this episode
        fish_name = env.np_random.choice(available_fish) if available_fish else None
        env.fish_name = fish_name

        observations = []
        actions = []
        rewards = []

        obs = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Take random actions or implement your policy here
            action = env.np_random.choice([0, 1])
            next_obs, reward, done, info = env.step(action)

            # Store data
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            step += 1

            if render:
                time.sleep(0.01)  # Slow down for visualization

        # Create episode data dictionary
        episode_data = {"fish_name": fish_name, "observations": np.array(observations), "actions": np.array(actions),
                        "rewards": np.array(rewards), "total_reward": sum(rewards), "length": len(rewards)}
        all_data.append(episode_data)

        print(f"Episode {episode + 1}/{episodes}, Fish: {fish_name}, "
              f"Reward: {episode_data['total_reward']:.2f}, Steps: {step}")

    env.close()
    return all_data


if __name__ == "__main__":
    # Create default fish.json if it doesn't exist
    create_default_fish_file()

    # Run the environment with human input for testing
    env = FishingMinigameEnv(render_mode="human")

    # Print available fish
    print("Available fish:", env.get_available_fish())


    # Add reset key binding
    def on_reset(event):
        if env.done and event.keysym == 'r':
            print("Resetting game...")
            # Choose a random fish for variety
            env.fish_name = env.np_random.choice(env.get_available_fish()) if env.fish_data else None
            env.reset()


    env.root.bind('<KeyPress-r>', on_reset)

    # Start the game loop manually
    obs = env.reset()

    while True:
        try:
            if env.done:
                # Wait for reset key input, but keep updating UI
                env.root.update()
                time.sleep(0.016)
                continue

            # Use the button_pressed state from key/mouse events
            action = FishingMinigameEnv.ACTION_PRESS if env.button_pressed else FishingMinigameEnv.ACTION_NONE
            obs, reward, done, info = env.step(action)

            # Update the UI
            env.root.update()
            time.sleep(0.016)  # ~60 FPS

        except tk.TclError:
            # Window was closed
            break
        except KeyboardInterrupt:
            break

    env.close()
