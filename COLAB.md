# Running 8-D Dueling DQN Training on Google Colab

The official competition architecture is an **8-D Dueling Double DQN** (~13.7k parameters, ~56 KB ONNX).
It runs fast on free Google Colab T4 GPUs and can even train on CPU.

| Configuration | Per-episode wall time | 5,000-episode total |
|---|---|---|
| CPU (8 threads) | ~0.5 - 1.0 second | ~45 - 60 minutes |
| **Colab T4 GPU (free tier)** | **~0.15 - 0.25 second** | **~15 - 20 minutes** |

Use the provided `colab_training.ipynb` to train the model, export it to `.onnx`, and download it directly for submission.

---

## Quick start (Colab)

1. Open `colab_training.ipynb` in GitHub / Google Colab.
2. **Runtime → Change runtime type → T4 GPU** (or CPU if GPU limits are reached).
3. Run all cells in order:
   - **Cell 1**: Clones the repo (`master` branch) and installs dependencies.
   - **Cell 2**: Verifies the GPU and the 8-D model architecture contract.
   - **Cell 3**: (Optional) Mounts Google Drive so checkpoints survive disconnections.
   - **Cell 4**: Trains the model via `python main.py --episodes 5000 --save-every 500`.
   - **Cell 5**: Exports the best checkpoint to `my_model.onnx`, verifies the ONNX graph against PyTorch, and downloads it.
4. Drag and drop `my_model.onnx` into the competition website (`/submit` or `/play`)!

---

## What each cell does

**Cell 1 — Setup.** Clones `master`, installs `torch`, `numpy`, `matplotlib`, `onnx`, `onnxruntime`, and `pygame`.

**Cell 2 — Verify Architecture.** Confirms CUDA availability and checks that:
- `OBS_DIM == 8` in `environment.py`
- `DQNAgent` and `DuelingDQN` match the ~13.7k parameter spec (~56 KB float32 ONNX).

**Cell 3 — (Optional) Mount Drive.** Saves checkpoints to `/content/drive/MyDrive/stardew-fishing-models` so they survive browser refreshes or runtime restarts.

**Cell 4 — Train.** Runs `python main.py --episodes {NUM_EPISODES} --save-every {SAVE_EVERY}` on 4 parallel vectorized environments.
- Logs evolution JSON artefacts and metrics CSVs.
- Progress prints every 100 episodes.

**Cell 5 — Export to ONNX & Download.**
- Locates the latest `.pth` checkpoint.
- Runs `python export_onnx.py <ckpt> --output my_model.onnx`.
- Verifies graph tolerance (`atol=1e-5`).
- Automatically triggers a browser download of `my_model.onnx` and latest training logs.

**Cell 6 — (Optional) Resume / Fine-Tune.** Allows uploading a `.pth` file and running additional training:
`python main.py --checkpoint <uploaded.pth> --episodes 2000 --eps-start 0.2`.

---

## Model Contract for Competition

The competition evaluator on the website expects:
- **Format:** `.onnx`
- **Input:** `state` with shape `[1, 8]` (`float32`)
- **Output:** `q_values` with shape `[1, 2]` (`float32`)
- **Max File Size:** ≤ 5 MB (baseline model is ~56.8 KB)
- **Target Latency:** ≤ 16 ms / step (60 FPS)
