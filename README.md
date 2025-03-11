Below is the reformatted Markdown document with improved structure, clarity, and consistent code blocks:

---

# Stardew Valley Fishing Minigame Double DQN

## Overview
This project implements a Deep Reinforcement Learning agent to master a fishing minigame in Stardew Valley. Using a Double Deep Q-Network (DDQN) architecture, the agent learns to time button presses perfectly to catch various fish with different movement behaviors and difficulty levels.

## Technical Features
- **Double DQN Implementation**: Reduces Q-value overestimation bias for more accurate decision making.
- **Advanced Network Architecture**: Utilizes a 3-layer neural network (128-128-64) with optimized weight initialization.
- **Curriculum Learning**: Gradually introduces harder fish types as the agent improves.
- **Time-Aware State Design**: Incorporates time as a state dimension for better strategic decisions.
- **CUDA Acceleration**: GPU-optimized tensor operations for faster training.

## Installation

Clone the repository and install the dependencies:

```bash
# Clone repository
git clone https://github.com/keethesh/StardewValleyFishingAI.git
cd StardewValleyFishingAI

# Install dependencies
pip install torch numpy matplotlib tkinter
```

## Usage

### Training a New Model
To train a new model, set `train_new_model = True` in the main block and run:

```bash
python main.py
```

### Using a Pre-trained Model
To use a pre-trained model:
1. Set `train_new_model = False` in the main block.
2. Ensure the model path is correctly specified.
3. Run:

```bash
python main.py
```

### Interactive Mode
After evaluation, the program enters interactive mode, where you can:
- Select specific fish to watch the agent catch.
- Choose random fish.
- Observe performance across different fish behaviors.

## Fish Behaviors
The agent learns to handle five distinct fish behaviors:
- **Mixed**: Combination of movement patterns.
- **Dart**: Quick, sudden movements requiring fast reactions.
- **Smooth**: Gradual, predictable movements.
- **Sinker**: Downward-biased movement.
- **Floater**: Upward-biased movement.

## Training Process
1. The agent starts with simpler fish (difficulty ≤ 40).
2. As performance improves, it progresses to medium difficulty (≤ 70).
3. Finally, it tackles the hardest fish (≤ 100).
4. Early stopping occurs after a consistent 98%+ success rate.

## Model Details
- **State Space**: 11 dimensions, including positions, speeds, fish characteristics, and time.
- **Action Space**: 2 actions (press/release button).
- **Network Architecture**: 3 hidden layers with sizes 128, 128, and 64 neurons respectively, using ReLU activations.
- **Learning Algorithm**: Double DQN with soft target updates.

## Best Models
Pre-trained models are saved in the "Best models" folder from multiple training runs. Each model represents a different training session with various hyperparameters and optimizations.

## Results
- **Overall Success Rate**: 95% across all fish types.
- **Smooth-Movement Fish**: 92.7% success.
- **Dart-Movement Fish**: 78.4% success.
- **Higher Difficulty Fish**: Near 100% success.

## Acknowledgments
This project demonstrates the application of recent reinforcement learning techniques to a timing-based game, showcasing how curriculum learning and network architecture choices can significantly impact learning efficiency.

---