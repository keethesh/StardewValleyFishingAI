# Stardew Fishing AI — Companion Site

Next.js app for the video companion: play the minigame, scrub training stages, and submit ONNX models.

## Quick start

```bash
cd competition-website
npm install
npm run copy-wasm
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Local Next.js server |
| `npm run build` | Production build |
| `npm run copy-wasm` | Copy `onnxruntime-web` WASM into `public/wasm/` |
| `npm run parity` | TS side of Python↔TS physics parity |

## Routes

| Path | Role |
|---|---|
| `/` | Landing + embedded demo |
| `/play` | Human vs AI (baseline = ep3500) |
| `/evolution` | Stage selector + telemetry |
| `/submit` | Upload `.onnx` → **official** server eval |
| `/rules` | Contract + scoring |

## Model contract

- Input: `state`, `float32`, shape `[1, 8]`
- Output: `q_values`, `float32`, shape `[1, 2]`
- Max size: 5 MB
- Latency target: ≤ 16 ms / step

Published baseline: `public/models/baseline.onnx` (episode 3500).

Stage snapshots: `public/models/stages/ep{20,500,1500,3500}.onnx`  
Traces: `public/data/evolution/episode_*.json`

## Scoring

| Mode | Where | What |
|---|---|---|
| Preview | Browser (`lib/evaluate-model.ts`) | Fixed fish list + fixed seeds — verify architecture |
| **Official** | Browser (`lib/simulation.ts`) | **All fish × 3 seeds**, fresh `runSeed` each submit. Evaluated client-side via WebAssembly in ~1.5s, then saved to Cloudflare KV. |

Leaderboard stores the official score (0–1). Training-repo `eval_score` uses the same formula with **public fixed seeds** — it is not the contest metric.

## Deploy on Cloudflare Workers

The site deploys to **Cloudflare Workers with Workers Static Assets** and native **Cloudflare KV**:

```bash
npm run deploy
```
