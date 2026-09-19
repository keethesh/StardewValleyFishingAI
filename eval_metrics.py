"""Training-only checkpoint evaluation.

Greedy policy over the full fish catalog. Not a competition metric —
just a stable signal that the policy is actually improving.
"""

from __future__ import annotations

import json
import os
from typing import Any

from environment import FishingMinigameEnv

EVAL_SEEDS_PER_FISH = 3
EVAL_BASE_SEED = 10_000
EVAL_MAX_T = 2000

_FISH_JSON = os.path.join(os.path.dirname(__file__), "data", "fish.json")


def load_eval_fish(path: str = _FISH_JSON) -> list[dict[str, Any]]:
    """Full catalog, sorted for stable seed assignment across runs."""
    with open(path, encoding="utf-8") as f:
        fish = json.load(f)
    return sorted(
        fish,
        key=lambda x: (str(x.get("behaviour", "")).lower(), float(x["difficulty"]), x["name"]),
    )


# Resolved at import so CLI / callers see the list; re-load inside evaluate if needed.
EVAL_FISH: list[dict[str, Any]] = load_eval_fish()


def evaluate_checkpoint(
    agent,
    fish_list: list[dict[str, Any]] | None = None,
    seeds_per_fish: int = EVAL_SEEDS_PER_FISH,
    base_seed: int = EVAL_BASE_SEED,
    max_t: int = EVAL_MAX_T,
) -> dict[str, Any]:
    """Run greedy eval on every fish. Returns eval_score in [0, 1] plus breakdowns."""
    fish_list = fish_list if fish_list is not None else load_eval_fish()
    env = FishingMinigameEnv(render_mode=None, augment_fish=False)

    per_fish: list[dict[str, Any]] = []
    by_behaviour: dict[str, dict[str, int]] = {
        b: {"attempts": 0, "successes": 0}
        for b in ("sinker", "dart", "smooth", "mixed", "floater")
    }

    weighted_sum = 0.0
    weight_total = 0.0
    catches = 0
    episodes = 0

    for fish_idx, fish in enumerate(fish_list):
        name = fish["name"]
        difficulty = float(fish["difficulty"])
        behaviour = str(fish["behaviour"]).lower()
        seed_scores: list[float] = []
        seed_caught: list[bool] = []

        for run in range(seeds_per_fish):
            seed = base_seed + fish_idx * seeds_per_fish + run
            env.fish_name = name
            state = env.reset(seed=seed)
            done = False
            steps = 0
            while not done and steps < max_t:
                action = agent.act(state, eps=0.0)
                state, _, done, _ = env.step(action)
                steps += 1

            caught = bool(env.distanceFromCatching >= 1.0)
            seed_caught.append(caught)
            seed_scores.append(difficulty if caught else 0.0)
            catches += int(caught)
            episodes += 1
            if behaviour in by_behaviour:
                by_behaviour[behaviour]["attempts"] += 1
                if caught:
                    by_behaviour[behaviour]["successes"] += 1

        per_fish_mean = sum(seed_scores) / seeds_per_fish
        weighted_sum += per_fish_mean
        weight_total += difficulty
        per_fish.append(
            {
                "name": name,
                "difficulty": difficulty,
                "behaviour": behaviour,
                "catch_rate": sum(seed_caught) / seeds_per_fish,
                "mean_weighted": per_fish_mean,
            }
        )

    env.close()

    eval_score = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    eval_catch_rate = catches / episodes if episodes else 0.0
    eval_by_behaviour = {
        b: (d["successes"] / d["attempts"] if d["attempts"] else 0.0)
        for b, d in by_behaviour.items()
    }

    hard = [f for f in per_fish if f["difficulty"] >= 70]
    hard_num = sum(f["mean_weighted"] for f in hard)
    hard_den = sum(f["difficulty"] for f in hard)
    eval_score_hard = (hard_num / hard_den) if hard_den else 0.0

    return {
        "eval_score": float(eval_score),
        "eval_catch_rate": float(eval_catch_rate),
        "eval_score_hard": float(eval_score_hard),
        "eval_by_behaviour": eval_by_behaviour,
        "per_fish": per_fish,
        "n_fish": len(fish_list),
        "episodes": episodes,
        "seeds_per_fish": seeds_per_fish,
    }


def format_eval_summary(result: dict[str, Any]) -> str:
    rates = result["eval_by_behaviour"]
    beh = ", ".join(f"{k}:{v:.0%}" for k, v in rates.items())
    n = result.get("n_fish", len(result.get("per_fish", [])))
    return (
        f"eval_score={result['eval_score']:.3f} "
        f"catch={result['eval_catch_rate']:.0%} "
        f"hard={result['eval_score_hard']:.3f} "
        f"fish={n} "
        f"[{beh}]"
    )


if __name__ == "__main__":
    import argparse

    import torch
    import torch.nn as nn

    from environment import OBS_DIM

    class _Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(OBS_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
            )
            self.value = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            self.advantage = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, x):
            h = self.feature(x)
            v = self.value(h)
            a = self.advantage(h)
            return v + (a - a.mean(dim=1, keepdim=True))

    class _Agent:
        def __init__(self, path: str):
            ckpt = torch.load(path, map_location="cpu")
            self.q = _Q()
            key = "q_state_dict" if "q_state_dict" in ckpt else "local_state_dict"
            self.q.load_state_dict(ckpt[key] if key in ckpt else ckpt)
            self.q.eval()

        def act(self, state, eps=0.0):
            with torch.no_grad():
                q = self.q(torch.FloatTensor(state).unsqueeze(0))
                return int(q.argmax(dim=1).item())

    parser = argparse.ArgumentParser(description="Run training checkpoint eval_score")
    parser.add_argument("checkpoint", type=str, help="Path to episode_*.pth")
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only print fish with catch_rate < 1",
    )
    args = parser.parse_args()

    result = evaluate_checkpoint(_Agent(args.checkpoint))
    print(format_eval_summary(result))
    rows = result["per_fish"]
    if args.failures_only:
        rows = [r for r in rows if r["catch_rate"] < 1.0]
    for row in rows:
        print(
            f"  {row['name']:20s} d={row['difficulty']:3.0f} "
            f"{row['behaviour']:7s} catch={row['catch_rate']:.0%}"
        )
