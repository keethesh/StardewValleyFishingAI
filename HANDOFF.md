# Handoff — RL Architecture, Learning Evolution, and Website Plan

**Purpose of this document:** give a fresh session everything it needs to (a) rebuild the
Next.js website from scratch, (b) confirm/finalise the RL architecture, and (c) build the
"evolution of the AI" showcase.

**Project context:** Stardew Valley fishing minigame, solved with reinforcement learning.
Deliverables are a YouTube video, a web demo, and a community model-training competition.

---

## 0. Current state of the repo (locked 2026-09-19)

| Thing | Where | Reality |
|---|---|---|
| Env | `environment.py` | **8-D** observation + Mulberry32 `PortableRNG` |
| Training | `main.py` | **Dueling Double DQN** + 3-step returns |
| Export | `export_onnx.py` | `[1,8]→[1,2]`, ~55 KB, ORT-verified |
| Baseline | `competition-website/public/models/baseline.onnx` | **ep3500** |
| Site physics | `competition-website/lib/game-logic.ts` | Line-by-line port; parity **200/200** |
| Checkpoint metric | `eval_metrics.py` | Full catalog × 3 **fixed** seeds → `eval_score` |
| Official score | `POST /api/evaluate` | Full catalog × 3 **fresh** seeds |
| Stages | `/evolution` | ep20 → ep500 → ep1500 → ep3500 |

> Historical note: earlier C51 / 14-D / 24-D stacks are obsolete. Ignore any `.pth` dated 2025-10-02.

---

## Part 1 — The ML architecture

### 1.1 Algorithm: Dueling Double DQN with 3-step returns

Genuine RL (no hard-coded heuristics), trains in ~20 min on free Colab, exports to a small
ONNX file, runs at 60 FPS in WASM.

Why Dueling fits this specific game:

- **State value `V(s)`** answers *"am I currently winning or losing?"* — meaningful at every
  frame regardless of action.
- **Advantage `A(s,a)`** answers *"does pressing right now beat releasing?"* — the only
  decision that matters in a binary-action, gravity-driven 1-D task.

```
Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )
```

Why **Double DQN**: the online net *selects* the best next action, the target net
*evaluates* it. Kills Q-value overestimation without any categorical projection machinery.

Why **N-step (N=3)**: credit assignment for "hold/tap correctly for a while" is delayed;
3-step returns propagate that signal ~3x faster.

### 1.2 State space (8-D)

```
[0] bobber_pos        / height        # fish vertical position
[1] bobber_vel        / speed_norm    # fish vertical velocity
[2] bar_pos           / height        # catch bar top edge
[3] bar_vel           / speed_norm    # catch bar velocity
[4] bar_height        / height
[5] (bar_center - bobber_pos) / height   # signed tracking error
[6] in_bar ? 1.0 : 0.0
[7] distanceFromCatching                 # [0, 1] catch progress
```

All values roughly normalised to `[-1, 1]`. No behaviour one-hot, no dead-reckoning.

> If 8-D turns out to underperform on `dart`/`floater` fish, the first feature to add back
> is **relative velocity** `(bobber_vel - bar_vel)`. That is the one signal that genuinely
> isn't recoverable from a single frame. Add it as `[8]`, don't go back to 24-D.

### 1.3 Network

```
[ 8-D state ]
      |
  Linear(8 -> 128) + ReLU
  Linear(128 -> 64) + ReLU
      |
  +---+---+
  |       |
Value   Advantage
(64->32)  (64->32)
 ReLU      ReLU
(32->1)   (32->2)
  |       |
  +---+---+
      |
  Q(s,a) = V + (A - mean(A))
      |
  [q_release, q_press]
```

Roughly **~14k parameters ≈ 55 KB as float32 ONNX** (the existing 14-D baseline is 157 KB,
which already fits comfortably in a browser and downloads instantly).

### 1.4 Training config

```python
gamma              = 0.99
lr                 = 2e-4          # Adam
batch_size         = 128
buffer_size        = 100_000       # uniform replay is fine at this scale
update_every       = 4
n_step             = 3
target_update_freq = 500           # hard update
eps_start          = 1.0
eps_end            = 0.02
eps_decay          = ~0.9995       # over ~10k episodes
grad_clip          = 1.0
```

Deliberately **not** included: NoisyNet, multi-head attention, residual projections, C51
atoms, PER, AMP, gradient accumulation, LR warmup. Each of these was added and each one
added a failure mode. See `IMPROVEMENTS.md` and commit `4d8ca98` — the C51 stack collapsed
its Q-value distribution to a span of 0.17 and could no longer distinguish press/release.

### 1.5 Reward shaping (current `environment.py` already does roughly this)

The existing reward hierarchy in `_calculate_reward` (lines 444-497) is good — keep it:

1. **Catch bonus dominates** (~`50 * difficulty * 0.02` + size + time bonus) — so catching
   always beats farming per-step reward.
2. **Catch-meter progress** is the primary per-step signal (`(dfc - prev) * 20.0`).
3. **In-bar bonus is intentionally weak** (`0.02` + Gaussian centering `0.05*exp(-4x²)`) —
   a precondition, not a goal.
4. **Velocity / movement penalties** (`-0.008`, `-0.005`) push toward smooth control.
5. Floater-specific stillness bonus/penalty.

> History worth keeping in the video: an earlier reward made "wiggle in the bar forever"
> worth +35 while an actual catch paid +14. The agent correctly learned to wiggle and never
> catch. That's a great segment — and it's why terminal reward must dominate.

### 1.6 ONNX export contract (publish this as the competition rule)

| Field | Value |
|---|---|
| Format | `.onnx` |
| Input | `state`, shape `[1, 8]`, `float32` |
| Output | `q_values`, shape `[1, 2]`, `float32` |
| Max size | 5 MB (baseline is ~160 KB — 5 MB is generous) |
| Latency | ≤ 16 ms per step (60 FPS) |

Export with `torch.onnx.export(..., dynamic_axes={'state': {0: 'batch'}, 'q_values': {0: 'batch'}})`.
Set `opset_version >= 17`. **Verify the exported graph** with `onnxruntime` on CPU against
the PyTorch output before shipping — a silent input-shape mismatch (exactly the 14 vs 24
bug already present in the repo) produces a model that loads fine and plays terribly.

---

## Part 2 — The evolution of the AI (the story)

This is the video's spine and the website's centrepiece. The agent discovers these
strategies *on its own*, in this order. The website should let a viewer scrub through them.

```
Ep 0 ─────── Ep 200 ─────── Ep 800 ─────── Ep 2,500 ─────── Ep 5,000
   |             |              |               |               |
[Flailing]  [Slamming]     [PWM Hover]    [Cushion Tap]   [Predictive]
```

### Stage 1 — The Panic Spammer (Ep 0–200)
- **Behaviour:** pins the bar to the ceiling, then releases and slams the floor. Pure
  epsilon exploration.
- **What it learns:** doing nothing = failure. First real lesson: stay off the extremes.
- **Win rate:** ~15% (only motionless `Carp`).

### Stage 2 — Discovery of Pulse Width Modulation (Ep 200–800)
- **The insight:** thrust is binary (0 or 1) but gravity is continuous. To *hover*, tap
  rapidly — e.g. 2 frames on, 2 frames off.
- **Behaviour shift:** erratic thrashing → a smooth rhythmic hum.
- **Unlocks:** `smooth` behaviour fish (Walleye, Salmon).
- **Win rate:** ~60%.
- **Website signal:** the "tap frequency" HUD metric spikes and stabilises here. This is the
  single most visually satisfying stage transition.

### Stage 3 — Cushion Tap / Bounce Cancellation (Ep 800–2,500)
- **The bottleneck:** the bar ricochets off the floor. High-speed impacts throw the fish out
  of the bar entirely.
- **The insight:** brake *before* contact — a single upward tap one or two frames before
  impact bleeds off downward momentum and lands soft.
- **Unlocks:** `sinker` fish (Octopus, Lava Eel types).
- **Win rate:** ~85%.
- **Website signal:** visible as a short "flare" on the thrust trace right before bottom
  contact.

### Stage 4 — Predictive Leading (Ep 2,500–5,000+)
- **The bottleneck:** on `dart` and `floater` fish, reacting to position is always ~100 ms
  too late. The bar chases and never catches up.
- **The insight:** the advantage stream starts weighting **relative velocity** over
  position. When the fish rockets upward, full thrust is applied *before* it leaves the bar.
  On floaters, it learns counter-intuitive patience — let the fish drift, don't overshoot.
- **Unlocks:** Legendaries (`Legend`, `Glacierfish`, `Crimsonfish`).
- **Win rate:** 95%+.

> **Calibration note:** the episode boundaries above are estimates and **must be verified
> against actual checkpoints** before they go in the video. `models/checkpoints/` has
> `episode_*.pth` every 500 episodes and `training_logs/graphs/episode_*.png` alongside —
> evaluate each checkpoint and read off where the behaviour actually changes. Do not narrate
> a stage boundary the data doesn't support.

### 2.1 Making the evolution *provable* (do this during training)

The story only works if it's backed by artefacts. Training must emit, per checkpoint:

1. **A short action trace** — `(timestep, action, bobber_pos, bar_center, in_bar)` for one
   episode against a fixed fish/seed. Drives the website replay and the thrust-trace chart.
2. **Per-behaviour win rates** — one number per behaviour class (`sinker`/`dart`/`smooth`/
   `mixed`/`floater`), so you can point at the exact episode where `sinker` jumped.
3. **Tap-frequency and mean-centering-error** rolling stats — these are the two metrics that
   visibly separate the four stages.

Traces are tiny (a few KB each) and can be bundled into the site as static JSON. This is far
cheaper than shipping every `.pth` checkpoint, and it's what actually shows the evolution.

---

## Part 3 — The website

**Rebuilding from scratch.** Suggested minimal surface:

```
/                    landing — video embed, "Play" CTA, leaderboard preview
/play                human vs AI, live, in-browser
/evolution           the four stages (this is the differentiator)
/compete             rules, contract, Colab link, upload
/leaderboard         ranked submissions
```

### 3.1 `/play` — human vs AI

- Canvas fishing game in TypeScript (`requestAnimationFrame`, fixed timestep).
- **The physics must be a port of `environment.py`, not a reimplementation.** Port the
  update loop line-by-line and add a test that steps both Python and TS with the same action
  sequence and compares positions. Any divergence invalidates every competition score.
- Baselines: `onnxruntime-web` with WASM + SIMD; try WebGPU, fall back silently.
- Same observation vector as training, computed identically.

### 3.2 `/evolution` — the showcase (highest value, build this first)

**a) Stage selector.** Four saved ONNX snapshots (`ep200`, `ep800`, `ep2500`, `ep5000`) run
on the *same fish and same seed*, side by side or one at a time. This is the money shot.

**b) Live telemetry HUD** next to the canvas:
- **Tap frequency (Hz)** — the PWM discovery.
- **Confidence ΔQ = Q(press) − Q(release)** — near zero means the model is on a knife's
  edge; large means it's certain. Genuinely interesting to watch.
- **Thrust trace** — rolling strip chart of the chosen action. The cushion tap appears as a
  discrete flare before floor contact.
- **Reward breakdown** — per-step contributions of the reward terms, so viewers see the
  agent is optimising something real.

**c) Ghost / race mode.** Same fish, same seed: human on the left, chosen checkpoint on the
right. Or ep800 vs ep5000 for a direct "how much better is it now" comparison.

**d) Training curves overlay.** Win rate + tap frequency + centering error on a shared
timeline, with a scrubber that seeks all three *and* the stage panel together. Drag the
scrubber and the fish animation seeks to the matching checkpoint. This is the feature that
makes the learning visible rather than asserted.

### 3.3 `/compete` — the competition

**Contract:** see §1.6. Restate it in one table on the page.

**Evaluation — must be server-side and hidden-seed:**

| | Where | Seeds | Purpose |
|---|---|---|---|
| Preview | Browser (WASM) | Public/fixed | Instant feedback, "does my model work?" |
| Official | Server | Random per run | The leaderboard that pays out |

Fixed-seed-only evaluation is trivially gameable: memorise the button sequence, submit a
lookup table. Random seeds on an unseen fish spread is the guardrail.

**Test suite:** reuse the curated 25-fish spread in `evaluate-model.ts` (Easy → Hard → all 10
Legendaries) but generate seeds server-side.

**Scoring:**
```
score = Σ over fish ( caught ? difficulty_weight : 0 ) × consistency_bonus
        + speed_bonus
```
Weight consistency over one-off heroics — run N episodes per fish and use the mean.

**Guardrails to publish:**
- No changing observation/action size.
- No modified physics constants that only exist in the submitter's fork.
- No hard-coded heuristics keyed to specific seeds.
- No dependency on helper code that vanishes once exported to ONNX.

**Prize payout flow:** winner submits training notebook → verify it actually trains a model
(not a disguised lookup table) → pay out. State this up front; it deters most of the nonsense.

**Starter pipeline:** one-click Colab that trains the 8-D Dueling DQN and ends in a
"Download my `.onnx`" cell. Test it end-to-end from a logged-out browser before launch.

### 3.4 Suggested stack

Next.js (App Router) + TypeScript + Tailwind + `onnxruntime-web`. Storage for
leaderboard/submissions: whichever is already wired in `competition-website/lib/`
(Supabase or Redis — both are present). Deploy on Vercel.

Do **not** rebuild the game logic or the evaluation scoring differently from the Python
side. Two implementations of the physics is the bug that quietly destroys the competition.

---

## Action checklist for the new session

1. **Decide the observation size** (recommendation: 8-D) and make env/trainer/exporter/site agree.
2. Port `environment.py` physics to TypeScript; add a Python↔TS parity test.
3. Train 8-D Dueling Double DQN → checkpoints + action traces + per-behaviour stats.
4. Evaluate every checkpoint; **verify** the four stage boundaries against the data.
5. Export 4 stage snapshots to ONNX; sanity-check each against PyTorch output.
6. Build `/evolution` first — it's the differentiator and the video's spine.
7. Build `/play`, then `/compete` + leaderboard.
8. Publish the Colab; verify it end-to-end from a clean browser session.
9. Record the video: failures (wiggle-exploit, Q-collapse) → discovery stages → Legend fish boss fight.

---

## Session progress (2026-09-19)

**Done:**
- [x] Locked observation to **8-D** across `environment.py`, `main.py`, `export_onnx.py`, and the website (`game-logic.ts`, eval/sim/submit/rules).
- [x] Replaced C51/NoisyNet stack with **Dueling Double DQN + 3-step returns** per §1.1–1.4 (~13.7k params).
- [x] Shared **Mulberry32** RNG (`portable_rng.py` ↔ `portable-rng.ts`) so physics can match bit-for-bit.
- [x] Ported `_update_game_logic` / `_calculate_reward` / `_get_observation` into `competition-website/lib/game-logic.ts`.
- [x] Parity test: `python tests/parity_check.py` — **200/200 steps PASS**.
- [x] Checkpoint artefacts: `training_logs/evolution/episode_*.json` (action trace + per-behaviour rates + tap Hz).
- [x] ONNX contract verified: `baseline.onnx` is **[1,8]→[1,2]**, ~55 KB, ORT vs PyTorch OK.

**Next (do these in order):**
1. ~~Run full training~~ → DONE
2. ~~Calibrate stages / export ONNX~~ → DONE (final baseline = **ep3500**)
3. ~~`/evolution` + `/play`~~ → DONE (`/evolution`, `/play`; nav linked)
4. ~~Official server eval~~ → DONE (`POST /api/evaluate`: **all fish × 3 fresh seeds**; submit uploads model)
5. Colab starter + video recording

### Published baseline decision
- **ep3500** is `public/models/baseline.onnx` (best `eval_score` ≈ 0.97; ep5000 was worse).
- Evolution final stage = ep3500 (not ep5000).
- Ice Pip / Scorpion Carp left soft on purpose for the challenge.

### Training metric (2026-09-19)

Replaced reliance on `Win_Rate_100` with **`eval_score`** (training-only, full catalog, **public fixed seeds**).
Official website leaderboard: **same formula, all fish × 3**, but **fresh seeds per submit**.

> Ignore `models/checkpoints/episode_5500.pth`…`episode_6500.pth` dated **2025-10-02** — those are from the old C51/24-D stack, not this 8-D Dueling run.

