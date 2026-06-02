# Running the C51 Training on Google Colab

The C51 Dueling architecture (256-256-128-64 + multi-head attention + 51
distributional atoms + NoisyNet + AdamW + grad accumulation) is **far too
heavy for CPU training**. Measured throughput:

| Configuration | Per-episode wall time | 12,000-episode total |
|---|---|---|
| CPU (8 threads, this machine) | ~25-45 seconds | **~26 hours** |
| **Colab T4 GPU (free tier)** | **~1-2 seconds** | **~2-3 hours** |
| Colab A100 GPU (Pro) | ~0.3 seconds | ~30-45 minutes |

Use the provided `colab_training.ipynb` to run the full 12,000-episode
training. Everything below is what that notebook does, in case you want
to replicate it manually or run on a different GPU host.

## Quick start (Colab)

1. Open `colab_training.ipynb` in GitHub
2. Click "Open in Colab" badge
3. **Runtime → Change runtime type → T4 GPU** (this is critical — without
   it, training takes days)
4. Run all cells in order
5. The last cell zips and downloads all checkpoints + logs

## What each cell does

**Cell 1 — Setup.** Clones the repo, checks out the `model-upgrade-v2`
branch (which has the C51 architecture), installs deps.

**Cell 2 — Verify GPU.** Confirms CUDA is available, imports the model
classes, prints parameter count (~414K for the C51 net).

**Cell 3 — (Optional) Mount Drive.** If you toggle `MOUNT_DRIVE=True`,
checkpoints save to `/content/drive/MyDrive/stardew-fishing-models` and
survive Colab disconnects.

**Cell 4 — Train.** Runs `python main.py` with `train_new_model=True`.
The script will:
- Train for up to 12,000 episodes with early stop at 98% success
- Save checkpoints every 500 episodes to `models/checkpoints/`
- Log metrics to `training_logs/training_metrics_<timestamp>.csv`
- Print progress every 100 episodes

**Cell 5 — Download.** Bundles the latest checkpoint, metrics CSV, and
milestone log into a zip and downloads it.

**Cell 6 — (Optional) Resume.** If you disconnected mid-training, upload
your last checkpoint and the script will continue from there.

## Expected training trajectory

Based on the proven Dueling DQN runs (2025-10-02, which used the same
24D state and similar architecture):

| Episode | Expected win rate | Notes |
|---|---|---|
| 1-50 | 0-10% | Epsilon 0.20 → ~0.18 (cosine decay), exploration-heavy |
| 100 | 30-50% | Easiest fish (difficulty ≤ 40) starting to be caught reliably |
| 300 | 50-70% | Easy fish mastered, medium unlocked |
| 500 | 70-85% | First saved checkpoint — should already be useful |
| 1000 | 85-95% | Medium fish mastered, hard unlocked |
| 2000+ | 95-99% | Convergence; early-stop may fire at 3 consecutive 98% eval rounds |
| 12000 | 95-99% | Final, or early-stopped earlier |

**If win rate is < 30% by episode 100**, something is wrong. Common causes:
- Forgot to switch to GPU runtime
- Training resumed from a broken checkpoint (start fresh)
- Random seed in environment was changed (don't change it)

## Why the architecture needs a GPU

The C51 model:
- Forward pass through 4 hidden layers + multi-head attention: ~5M FLOPs
- 51 distributional atoms × 2 actions = 102 output values
- N-step returns with n=3, plus prioritized replay
- Gradient accumulation (2 steps) and AdamW (extra state per param)
- 4 parallel envs stepping in lockstep, each contributing to the batch

On CPU (8 threads), the bottleneck is the matrix multiplies on the
attention block and the C51 head. On a T4 GPU, these run 10-20x faster
because the entire batch fits in VRAM and runs as a single fused kernel.

If you must train on CPU (e.g. no GPU access), use the smaller config
in `sanity_check_cpu.py` (hidden=[128,128,64], no attention, batch=64,
no grad accumulation) — gets ~1 sec/episode on CPU but is weaker
architecturally.

## Sanity check before committing to a long run

Before launching the full 12k episodes, run `sanity_check_cpu.py` to
verify the 3 critical fixes are in place:

```
python sanity_check_cpu.py
```

This takes ~60 seconds and confirms:
1. `state[19]` (predicted fish position) carries real signal
   (was dead due to a double-division bug, std ≈ 0)
2. The observation buffer is not shared between state and next_state
   (was a single mutable array, breaking TD learning)
3. Epsilon starts at 0.20 (was 0.05, 4x less exploration)
4. The C51 loss is finite and has variance (network is producing output)
5. The vectorized training loop runs end-to-end without NaN

If any of these fail, fix the issue before launching on Colab — a
12,000-episode run that crashes at episode 100 wastes 30+ minutes of GPU time.
