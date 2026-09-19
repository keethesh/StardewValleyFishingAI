# Stardew Valley Fishing AI

Dueling Double DQN agent for the Stardew Valley fishing minigame, plus a Next.js companion site for demos and a community challenge.

## Stack (locked)

| Piece | Choice |
|---|---|
| Observation | **8-D** (`environment.py` / `game-logic.ts`) |
| Algorithm | Dueling Double DQN + 3-step returns |
| Export | ONNX `[1,8] → [1,2]`, ~55 KB |
| Published baseline | `episode_3500` → `competition-website/public/models/baseline.onnx` |

Physics is shared: Python `PortableRNG` (Mulberry32) matches TypeScript `portable-rng.ts`. Parity: `python tests/parity_check.py`.

## Setup (training)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install torch numpy matplotlib onnx onnxruntime pygame
```

## Train

```bash
python main.py --episodes 5000
```

Checkpoints write every 500 episodes under `models/checkpoints/`. Each save also writes:
- action trace + metrics → `training_logs/evolution/episode_N.json`
- **`eval_score`** — greedy eval on the full fish catalog × 3 fixed seeds (`eval_metrics.py`)

```bash
# Score any checkpoint (training pulse; public fixed seeds)
python eval_metrics.py models/checkpoints/episode_3500.pth --failures-only
```

`Win_Rate_100` in the CSV is only a live training pulse. Prefer **`eval_score`** at checkpoints.

## Export ONNX

```bash
python export_onnx.py models/checkpoints/episode_3500.pth --output competition-website/public/models/baseline.onnx
```

## Companion site

See [`competition-website/README.md`](competition-website/README.md).

- `/play` — human vs baseline AI  
- `/evolution` — stage scrubber (ep20 → 500 → 1500 → 3500)  
- `/submit` — upload ONNX; **official** score is all fish × 3 **fresh** seeds (server)

## Project layout

```
environment.py          # Gym-like env, 8-D obs, rewards
main.py                 # Dueling Double DQN trainer
eval_metrics.py         # Checkpoint eval_score (full catalog)
export_onnx.py          # ONNX export + ORT verify
portable_rng.py         # Mulberry32 (parity with TS)
tests/parity_check.py   # Python ↔ TS physics parity
competition-website/    # Next.js demo + challenge
HANDOFF.md              # Architecture + session status
```

## Notes

- Old C51 / 14-D / 24-D checkpoints are obsolete; do not mix them with this stack.
- Ice Pip and Scorpion Carp remain soft on the published baseline on purpose (challenge headroom).
- `HANDOFF.md` has the full design rationale and checklist.
