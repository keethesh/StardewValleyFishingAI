import json
import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pygame

# Configure logging
logger = logging.getLogger(__name__)


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

    def __init__(self, render_mode="human", seed=None, fish_name=None, normalize_obs=False, augment_fish=True):
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

        # Data augmentation flag (randomize fish parameters slightly for generalization)
        self.augment_fish = augment_fish

        # Pygame rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.button_pressed = False  # Track button state for human play

        # Sprite storage
        self.sprites = {}
        self.sprites_loaded = False

        # For tracking ML training progress
        self.episode_reward = 0
        self.episode_length = 0

        # Observation normalization
        self.normalize_obs = normalize_obs
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        self.obs_sum = None
        self.obs_sum_sq = None

        # Cache normalized constants for better performance
        self._norm_constants = {
            'height': self.track_height,
            'max_fish_size': 20,  # Will be set in reset
            'speed_norm': 10.0,
            'accel_norm': 5.0,  # For acceleration normalization
            'max_timesteps': 2000
        }

        # Temporal feature tracking (for velocity/acceleration)
        self._state_history = deque(maxlen=3)
        self._prev_bobber_speed = 0.0
        self._prev_bar_speed = 0.0

        # Initialize Pygame if needed
        if render_mode == "human":
            self.setup_pygame()

        # Initialize the environment
        self.reset()

    def load_fish_data(self):
        """Load fish data from fish.json file."""
        try:
            with open("data/fish.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading fish.json: {e}")
            # Provide default fish data if file not found or invalid
            return [{"name": "Default Fish", "difficulty": 50, "behaviour": "mixed"}]

    def select_fish(self, fish_name=None):
        """Select a fish by name or randomly, with optional parameter augmentation."""
        if fish_name:
            # Find fish by name
            for fish in self.fish_data:
                if fish["name"].lower() == fish_name.lower():
                    selected_fish = fish.copy()
                    break
            else:
                # Fish not found, select random
                selected_fish = self.np_random.choice(self.fish_data).copy() if self.fish_data else {
                    "name": "Default Fish", "difficulty": 50, "behaviour": "mixed"}
        else:
            # Select random fish if not found or none specified
            if self.fish_data:
                selected_fish = self.np_random.choice(self.fish_data).copy()
            else:
                selected_fish = {"name": "Default Fish", "difficulty": 50, "behaviour": "mixed"}

        # Apply data augmentation if enabled (randomize difficulty slightly)
        if self.augment_fish:
            # Add random variation to difficulty (±10%)
            difficulty_variation = self.np_random.uniform(-0.10, 0.10)
            augmented_difficulty = selected_fish["difficulty"] * (1.0 + difficulty_variation)
            # Clamp to valid range [1, 110]
            selected_fish["difficulty"] = int(np.clip(augmented_difficulty, 1, 110))

        return selected_fish

    def get_motion_type(self, behaviour):
        """Convert behaviour string to motion type."""
        behaviour = behaviour.lower()
        return self.BEHAVIOR_TYPES.get(behaviour, 0)  # Default to mixed if unknown

    def load_sprites(self):
        """Load Stardew Valley sprites from individual asset files."""
        try:
            # Define sprite files
            sprite_files = {
                'background': 'assets/background.png',
                'fish_normal': 'assets/fish_normal.png',
                'fish_boss': 'assets/fish_boss.png',
                'catch_bar_top': 'assets/catch_bar_top.png',
                'catch_bar_mid': 'assets/catch_bar_mid.png',
                'catch_bar_bot': 'assets/catch_bar_bot.png',
                'handle': 'assets/handle.png'
            }

            # Load and scale each sprite
            UI_SCALE = 4.0  # Authentic 4x UI scale from Stardew Valley

            for name, file_path in sprite_files.items():
                try:
                    sprite_surface = pygame.image.load(file_path).convert_alpha()
                    original_width, original_height = sprite_surface.get_size()
                    scaled_width = int(original_width * UI_SCALE)
                    scaled_height = int(original_height * UI_SCALE)
                    scaled_surface = pygame.transform.scale(sprite_surface, (scaled_width, scaled_height))
                    self.sprites[name] = scaled_surface
                except Exception as e:
                    logger.warning(f"Could not load sprite '{name}' from {file_path}: {e}")
                    continue

            if len(self.sprites) > 0:
                logger.info(f"Loaded {len(self.sprites)} Stardew Valley sprites from assets/")
                return True
            else:
                logger.error("No sprites could be loaded from assets/")
                return False

        except Exception as e:
            logger.error(f"Critical error loading sprites: {e}")
            return False

    def setup_pygame(self):
        """Setup Pygame display and components."""
        pygame.init()
        pygame.font.init()

        # Create display with proper dimensions
        self.screen = pygame.display.set_mode((800, 700))
        pygame.display.set_caption("Stardew Valley Fishing Minigame")

        # Create clock for frame rate control
        self.clock = pygame.time.Clock()

        # Load font for text
        try:
            self.font = pygame.font.Font(None, 24)
        except:
            self.font = pygame.font.SysFont('Arial', 24)

        # UI constants for authentic Stardew Valley layout
        self.UI_SCALE = 4.0
        self.UI_BASE_X = 200
        self.UI_BASE_Y = 50

        # Load sprites
        self.sprites_loaded = self.load_sprites()

    def handle_pygame_events(self):
        """Handle Pygame events for input and window management"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.button_pressed = True
                elif event.key == pygame.K_ESCAPE:
                    return False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.button_pressed = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.button_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.button_pressed = False
        return True

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

        # Reset temporal tracking
        self._state_history.clear()
        self._prev_bobber_speed = 0.0
        self._prev_bar_speed = 0.0

        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        """Convert game state to ML-friendly observation vector with temporal features."""
        # Calculate accelerations (change in speed)
        bobber_acceleration = (self.bobberSpeed - self._prev_bobber_speed) / self._norm_constants['accel_norm']
        bar_acceleration = (self.bobberBarSpeed - self._prev_bar_speed) / self._norm_constants['accel_norm']

        # Update previous speeds for next calculation
        self._prev_bobber_speed = self.bobberSpeed
        self._prev_bar_speed = self.bobberBarSpeed

        # Calculate distance to bar center (helps agent know direction to move)
        bar_center = self.bobberBarPos + (self.bobberBarHeight / 2.0)
        distance_to_bar = (self.bobberPosition - bar_center) / self._norm_constants['height']

        # Build observation with temporal features
        obs = np.array([
            self.bobberPosition / self._norm_constants['height'],  # fish position
            self.bobberSpeed / self._norm_constants['speed_norm'],  # fish speed
            bobber_acceleration,  # NEW: fish acceleration
            self.bobberBarPos / self._norm_constants['height'],  # bar position
            self.bobberBarSpeed / self._norm_constants['speed_norm'],  # bar speed
            bar_acceleration,  # NEW: bar acceleration
            self.bobberBarHeight / self._norm_constants['height'],  # bar height
            distance_to_bar,  # NEW: signed distance to bar center
            float(self.bobberInBar),  # binary: fish in bar?
            self.distanceFromCatching,  # progress toward catching (0-1)
            self.fishSize / self._norm_constants['max_fish_size'],  # fish size
            self.difficulty / 100.0,  # difficulty
            float(self.motionType) / 4.0,  # motion type
            self.current_timestep / self._norm_constants['max_timesteps'],  # time progress
        ], dtype=np.float32)

        # Apply running normalization if enabled
        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        return obs

    def _normalize_observation(self, obs):
        """Apply running mean/std normalization to observations."""
        # Initialize statistics on first call
        if self.obs_mean is None:
            obs_dim = obs.shape[0]
            self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
            self.obs_std = np.ones(obs_dim, dtype=np.float32)
            self.obs_sum = np.zeros(obs_dim, dtype=np.float32)
            self.obs_sum_sq = np.zeros(obs_dim, dtype=np.float32)

        # Update running statistics
        self.obs_count += 1
        self.obs_sum += obs
        self.obs_sum_sq += obs ** 2

        # Calculate running mean and std
        self.obs_mean = self.obs_sum / self.obs_count
        variance = (self.obs_sum_sq / self.obs_count) - (self.obs_mean ** 2)
        self.obs_std = np.sqrt(np.maximum(variance, 1e-8))

        # Normalize observation
        normalized_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)

        return normalized_obs

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
            if not self.handle_pygame_events():
                done = True
            self._render_frame()
            if self.clock:
                self.clock.tick(60)

        return obs, reward, done, info

    def _calculate_reward(self, prev_distance):
        """Calculate reward based on current state and previous state with improved shaping."""
        # Progress reward: improvement in catching progress
        progress_reward = (self.distanceFromCatching - prev_distance) * 10.0

        # Reward for keeping fish in bar WITH centering bonus
        bar_center = self.bobberBarPos + (self.bobberBarHeight / 2.0)

        if self.bobberInBar:
            # Base reward for being in bar
            in_bar_reward = 0.1

            # NEW: Centering bonus - reward keeping fish centered in bar
            fish_position_in_bar = abs(self.bobberPosition - bar_center) / (self.bobberBarHeight / 2.0)
            centering_bonus = 0.15 * (1.0 - fish_position_in_bar)  # Higher bonus for center
            in_bar_reward += centering_bonus
        else:
            in_bar_reward = -0.05
            centering_bonus = 0.0

        # Proximity reward - guide bar toward fish when not in bar
        distance_to_fish = abs(bar_center - self.bobberPosition)
        normalized_distance = distance_to_fish / self.track_height
        proximity_reward = -0.02 * normalized_distance  # Closer is better

        # Velocity matching reward - encourage smooth control
        velocity_diff = abs(self.bobberSpeed - self.bobberBarSpeed)
        velocity_penalty = -0.005 * velocity_diff

        # Penalty for extreme movements (reduced weight)
        movement_penalty = -0.005 * abs(self.bobberBarSpeed)

        # Early progress bonus - combat sparse rewards early on
        early_bonus = 0.0
        if self.distanceFromCatching < 0.3 and self.bobberInBar:
            early_bonus = 0.05  # Extra encouragement in early stages

        # Time efficiency penalty - encourages faster catches
        time_penalty = -0.01

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

        return (progress_reward + in_bar_reward + proximity_reward + velocity_penalty +
                movement_penalty + early_bonus + time_penalty) * difficulty_factor

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

    def _get_red_to_green_lerp_color(self, value):
        """Calculates a color between red and green based on a 0.0-1.0 value."""
        r = int(min(2.0 - 2.0 * value, 1.0) * 255)
        g = int(min(2.0 * value, 1.0) * 255)
        return (r, g, 0)

    def _render_frame(self):
        """Render the current frame using authentic Stardew Valley sprites."""
        if self.screen is None:
            return

        # Clear screen with dark blue background
        self.screen.fill((50, 50, 80))

        # Use authentic sprite-based rendering if sprites are loaded
        if self.sprites_loaded:
            self._draw_fishing_ui_sprites()
        else:
            self._draw_fishing_ui_fallback()

        # Update display
        pygame.display.flip()

    def _draw_fishing_ui_sprites(self):
        """Draw the fishing UI using authentic Stardew Valley sprites."""
        # Draw the Bobber Bar Background
        if 'background' in self.sprites:
            bg_sprite = self.sprites['background']
            bg_x = self.UI_BASE_X + 70 - (bg_sprite.get_width() / 2)
            bg_y = self.UI_BASE_Y + 296 - (bg_sprite.get_height() / 2)
            self.screen.blit(bg_sprite, (bg_x, bg_y))

        # Draw the Green Catch Bar (Stretched from 3 parts)
        bar_x = self.UI_BASE_X + 64
        bar_y = self.UI_BASE_Y + 12 + self.bobberBarPos

        if all(key in self.sprites for key in ['catch_bar_top', 'catch_bar_mid', 'catch_bar_bot']):
            top_sprite = self.sprites['catch_bar_top']
            mid_sprite = self.sprites['catch_bar_mid']
            bot_sprite = self.sprites['catch_bar_bot']

            # Calculate middle section height
            original_top_height = 2 * self.UI_SCALE
            original_bot_height = 2 * self.UI_SCALE
            middle_height = self.bobberBarHeight - (original_top_height + original_bot_height)

            # Scale middle section to fill the gap
            mid_scaled = pygame.transform.scale(mid_sprite, (mid_sprite.get_width(), int(middle_height)))

            # Draw the three parts
            self.screen.blit(top_sprite, (bar_x, bar_y))
            self.screen.blit(mid_scaled, (bar_x, bar_y + top_sprite.get_height()))
            self.screen.blit(bot_sprite, (bar_x, bar_y + top_sprite.get_height() + mid_scaled.get_height()))

        # Draw the Fish Icon
        is_boss_fish = self.difficulty >= 80
        fish_sprite_key = 'fish_boss' if is_boss_fish else 'fish_normal'

        if fish_sprite_key in self.sprites:
            fish_sprite = self.sprites[fish_sprite_key]
            fish_x = self.UI_BASE_X + 64 + 18 - (fish_sprite.get_width() / 2)
            fish_draw_y = self.UI_BASE_Y + 12 + 24 + self.bobberPosition - (fish_sprite.get_height() / 2)
            self.screen.blit(fish_sprite, (fish_x, fish_draw_y))

        # Draw the Catch Percentage Indicator
        bar_total_height_pixels = 580
        bar_width_pixels = 16

        indicator_height = int(bar_total_height_pixels * self.distanceFromCatching)
        indicator_y = self.UI_BASE_Y + 4 + (bar_total_height_pixels - indicator_height)
        indicator_x = self.UI_BASE_X + 124

        color = self._get_red_to_green_lerp_color(self.distanceFromCatching)
        pygame.draw.rect(self.screen, color, (indicator_x, indicator_y, bar_width_pixels, indicator_height))

        # Draw Fish Info Text
        text_x = self.UI_BASE_X + 160
        fish_text = self.font.render(f"Fish: {self.current_fish['name']}", True, (255, 255, 255))
        self.screen.blit(fish_text, (text_x, self.UI_BASE_Y))

        difficulty_text = self.font.render(f"Difficulty: {self.difficulty}", True, (255, 255, 255))
        self.screen.blit(difficulty_text, (text_x, self.UI_BASE_Y + 25))

        progress_text = self.font.render(f"Progress: {self.distanceFromCatching:.1%}", True, (255, 255, 255))
        self.screen.blit(progress_text, (text_x, self.UI_BASE_Y + 50))

        # Game state info
        if self.done:
            font_large = pygame.font.Font(None, 48)
            result = "YOU WIN!" if self.distanceFromCatching >= 1.0 else "YOU LOSE!"
            result_text = font_large.render(result, True, (255, 255, 255))
            result_rect = result_text.get_rect(center=(400, 350))
            self.screen.blit(result_text, result_rect)
        else:
            controls_text = self.font.render("Hold SPACE to pull", True, (255, 255, 255))
            self.screen.blit(controls_text, (text_x, self.UI_BASE_Y + 75))

    def _draw_fishing_ui_fallback(self):
        """Fallback UI drawing when sprites aren't available."""
        TRACK_X = 100
        TRACK_Y = 50
        TRACK_WIDTH = 50
        TRACK_HEIGHT = 568

        # Colors
        COLOR_TRACK = (48, 110, 156)
        COLOR_PLAYER_BAR = (125, 225, 80)
        COLOR_FISH = (227, 158, 48)
        COLOR_PROGRESS_BG = (100, 100, 100)

        # Draw the main bobber bar track
        pygame.draw.rect(self.screen, COLOR_TRACK, (TRACK_X, TRACK_Y, TRACK_WIDTH, TRACK_HEIGHT))

        # Draw player bar (catch area)
        bar_y = TRACK_Y + self.bobberBarPos
        bar_height = self.bobberBarHeight
        player_bar_rect = pygame.Rect(TRACK_X, bar_y, TRACK_WIDTH, bar_height)

        bar_color = COLOR_PLAYER_BAR if self.bobberInBar else (255, 165, 0)
        pygame.draw.rect(self.screen, bar_color, player_bar_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), player_bar_rect, 2)

        # Draw fish
        fish_y = TRACK_Y + self.bobberPosition
        fish_height = 20
        fish_rect = pygame.Rect(TRACK_X, fish_y, TRACK_WIDTH, fish_height)
        pygame.draw.rect(self.screen, COLOR_FISH, fish_rect)

        # Draw progress bar
        progress_bar_x = TRACK_X + TRACK_WIDTH + 20
        progress_bar_height = TRACK_HEIGHT

        pygame.draw.rect(self.screen, COLOR_PROGRESS_BG, (progress_bar_x, TRACK_Y, 30, progress_bar_height))

        fill_height = progress_bar_height * self.distanceFromCatching
        fill_color = self._get_red_to_green_lerp_color(self.distanceFromCatching)

        pygame.draw.rect(self.screen, fill_color,
                         (progress_bar_x, TRACK_Y + progress_bar_height - fill_height, 30, fill_height))

        pygame.draw.rect(self.screen, (255, 255, 255), (progress_bar_x, TRACK_Y, 30, progress_bar_height), 2)

        # Text info
        text_x = TRACK_X + TRACK_WIDTH + 70
        fish_text = self.font.render(f"Fish: {self.current_fish['name']}", True, (255, 255, 255))
        self.screen.blit(fish_text, (text_x, TRACK_Y))

        if self.done:
            font_large = pygame.font.Font(None, 48)
            result = "YOU WIN!" if self.distanceFromCatching >= 1.0 else "YOU LOSE!"
            result_text = font_large.render(result, True, (255, 255, 255))
            result_rect = result_text.get_rect(center=(400, 350))
            self.screen.blit(result_text, result_rect)

    def set_render_mode(self, mode: str):
        """Change the render mode and setup/cleanup accordingly."""
        if mode == self.render_mode:
            return

        # Clean up old mode
        if self.render_mode == "human" and self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

        # Setup new mode
        self.render_mode = mode
        if mode == "human":
            self.setup_pygame()

    def close(self):
        """Close the environment and clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

    def get_available_fish(self):
        """Return list of available fish names."""
        return [fish["name"] for fish in self.fish_data]


# Create a default fish.json file if it doesn't exist
def create_default_fish_file():
    """Create a default fish.json file if it doesn't exist."""
    if not os.path.exists("data/fish.json"):
        default_fish = [{"name": "Pufferfish", "difficulty": 80, "behaviour": "floater"},
                        {"name": "Salmon", "difficulty": 50, "behaviour": "mixed"},
                        {"name": "Octopus", "difficulty": 95, "behaviour": "sinker"},
                        {"name": "Trout", "difficulty": 30, "behaviour": "mixed"},
                        {"name": "Shark", "difficulty": 90, "behaviour": "dart"}]

        try:
            os.makedirs("data", exist_ok=True)
            with open("data/fish.json", "w") as f:
                json.dump(default_fish, f, indent=2)
            logger.info("Created default data/fish.json file")
        except Exception as e:
            logger.error(f"Error creating fish.json: {e}")


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
    print("Controls: SPACE or mouse click to pull up, release to let down")
    print("Press ESC to quit, R to reset when done")

    # Start the game loop manually
    obs = env.reset()

    while True:
        try:
            # Handle pygame events
            if not env.handle_pygame_events():
                break

            # Check for reset key (R)
            keys = pygame.key.get_pressed()
            if env.done and keys[pygame.K_r]:
                print("Resetting game...")
                env.fish_name = env.np_random.choice(env.get_available_fish()) if env.fish_data else None
                env.reset()
                continue

            if env.done:
                # Wait for reset key input
                env.clock.tick(60)
                continue

            # Use the button_pressed state from key/mouse events
            action = FishingMinigameEnv.ACTION_PRESS if env.button_pressed else FishingMinigameEnv.ACTION_NONE
            obs, reward, done, info = env.step(action)

            # Maintain frame rate
            env.clock.tick(60)

        except KeyboardInterrupt:
            break

    env.close()
