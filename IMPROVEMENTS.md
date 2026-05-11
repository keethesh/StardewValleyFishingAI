# Comprehensive Model Upgrade — C51 Distributional DQN

**Branch:** `model-upgrade-worktree` (worktree at `../StardewValleyML-improved`)
**Date:** 2026-05-11

---

## What Changed

### 1. Network Architecture: DuelingDQN → C51 Distributional DQN (+Attention +Residuals)

| Feature | Before | After |
|---|---|---|
| Algorithm | Dueling DQN | C51 Distributional DQN (51 atoms) |
| Hidden layers | [128, 128, 64] | [256, 256, 128, 64] |
| Activation | ReLU | GELU |
| Exploration | Epsilon-greedy | NoisyNet (learnable noise) |
| Residual connections | No | Yes (skip connections in shared layers) |
| Multi-head attention | No | Yes (4 heads, embed_dim=128/64) |
| Weight init | Kaiming normal | Kaiming normal (same) |
| Gradient clipping | 1.0 | 1.0 (same) |
| L2 regularization | No | AdamW weight_decay=1e-5 |

### 2. State Representation: 14D → 24D

| Feature | Before | After |
|---|---|---|
| Base temporal features | 14D (pos, speed, accel, in_bar, progress, etc.) | Same 14D |
| Behavior one-hot | No | +5D (sinker, dart, smooth, mixed, floater) |
| Predicted fish pos (1-step) | No | +1D |
| Predicted fish pos (3-step) | No | +1D |
| Position error | No | +1D (distance from optimal bar center) |
| Frame history (pos deltas) | No | +2D (bobber delta, bar delta) |
| **Total** | **14D** | **24D** |

### 3. Reward Function Enhancements

| Component | Before | After |
|---|---|---|
| Centering bonus | Linear 0.15× | Gaussian 0.25×exp(-4x²) |
| Floater-specific reward | None | +0.3 for smooth idle, -0.1×speed for jitter |
| Velocity penalty | -0.005 | -0.008 (stronger smoothness signal) |
| Proximity reward | -0.02 | -0.025 |

### 4. Training Algorithm Improvements

| Feature | Before | After |
|---|---|---|
| Algorithm | Double DQN + Dueling + PER + N-step | C51 + NoisyNet + Dueling + PER + N-step |
| Support range | N/A | [-20, +20] over 51 atoms |
| Loss function | MSE (Q-values) | Cross-entropy (return distribution) |
| Optimizer | Adam | AdamW (with weight decay) |
| LR scheduler | CosineAnnealing (T_max=10000) | CosineAnnealing (T_max=50000) |
| LR warmup | No | Yes (1000 steps) |
| Buffer size | 100,000 | 150,000 |
| Target update freq | 1000 steps | 500 steps (more stable for C51) |
| Gradient accumulation | No | Yes (2 steps) |
| Beta annealing | 0.001 increment | 0.001 increment (same) |

### 5. Floater-Specific Curriculum

- **Trigger:** After reaching 80% overall win rate
- **Duration:** 1000 dedicated floater episodes
- **What it does:** Forces 100% floater fish for 1000 episodes to specifically address the 50% floater success rate
- **Reward shaping:** +0.3 bonus for smooth, minimal-action bar control on floaters; penalty for jittering

### 6. Environment Changes

- **24D observation space** with behavior encoding, prediction features, position error, frame history
- **Gaussian centering bonus** (smoother peak than linear ramp)
- **Floater-specific reward** components (reward idle, penalize jitter)
- **Frame deltas** in state for velocity history

---

## Expected Impact

| Metric | Before | Expected After | Improvement Driver |
|---|---|---|---|
| Overall WR | 90-91% | 95-98% | C51 + NoisyNet + bigger net |
| Floater WR | 50% | 80-90% | Floater curriculum + reward + one-hot encoding |
| Dart WR | 78.4% | 85-92% | Attention + prediction features |
| Smooth WR | 92.7% | 95-99% | Residual connections |
| Catch speed | 50+ steps | faster | Better centering rewards |
| Training stability | Moderate | High | C51 distributional loss + grad accum |

---

## How to Train

```bash
# In the worktree:
cd ../StardewValleyML-improved
source .venv/Scripts/activate
python main.py
```

The `train_new_model = True` flag is set by default. Training runs for up to 12,000 episodes with early stopping at 98%+ success rate (3 consecutive evaluation rounds).

---

## Summary of Architectural Improvements

```
Before (Dueling DQN):              After (C51 DQN):
┌─────────────┐                   ┌──────────────────┐
│ 14D State   │                   │ 24D State        │
├─────────────┤                   ├──────────────────┤
│ ReLU(128)   │                   │ GELU(256)        │
│ ReLU(128)   │                   │ GELU(256) + Skip │
│ ReLU(64)    │                   │ GELU(128) + Skip │
├─────────────┤                   │ GELU(64)  + Skip │
│ V   A       │                   ├──────────────────┤
├─────────────┤                   │ MultiheadAttn(4) │
│ Q-values    │                   ├──────────────────┤
└─────────────┘                   │ V_logits  A_logits│
                                  ├──────────────────┤
                                  │ Softmax → Dist   │
                                  │ Q = Σ(p_i × z_i) │
                                  └──────────────────┘
Learnable noise:    NoisyNet layers replace epsilon-greedy
Value distribution: C51 learns full probability over 51 atoms
