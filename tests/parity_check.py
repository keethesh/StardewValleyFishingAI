"""Generate a Python physics fixture and compare against the TypeScript port.

Usage:
  python tests/parity_check.py              # dump fixture + run TS check via npx tsx
  python tests/parity_check.py --dump-only  # only write fixture JSON
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment import FishingMinigameEnv
from portable_rng import PortableRNG

FIXTURE_PATH = os.path.join(ROOT, "competition-website", "scripts", "parity-fixture.json")
TS_CHECK = os.path.join(ROOT, "competition-website", "scripts", "parity-check.ts")


def dump_fixture(seed: int = 12345, fish_name: str = "Carp", steps: int = 200) -> dict:
    env = FishingMinigameEnv(
        render_mode=None, seed=seed, fish_name=fish_name, augment_fish=False
    )
    env.fish_name = fish_name
    obs = env.reset(seed=seed)

    # Deterministic action sequence (press pattern) — not random, so RNG stays in physics
    actions = [1 if (i % 4) < 2 else 0 for i in range(steps)]

    frames = [
        {
            "t": 0,
            "bobberPosition": env.bobberPosition,
            "bobberSpeed": env.bobberSpeed,
            "bobberTargetPosition": env.bobberTargetPosition,
            "bobberBarPos": env.bobberBarPos,
            "bobberBarSpeed": env.bobberBarSpeed,
            "floaterSinkerAcceleration": env.floaterSinkerAcceleration,
            "distanceFromCatching": env.distanceFromCatching,
            "bobberInBar": bool(env.bobberInBar),
            "obs": obs.tolist(),
            "reward": None,
        }
    ]

    for i, action in enumerate(actions):
        obs, reward, done, _ = env.step(action)
        frames.append(
            {
                "t": i + 1,
                "action": action,
                "bobberPosition": env.bobberPosition,
                "bobberSpeed": env.bobberSpeed,
                "bobberTargetPosition": env.bobberTargetPosition,
                "bobberBarPos": env.bobberBarPos,
                "bobberBarSpeed": env.bobberBarSpeed,
                "floaterSinkerAcceleration": env.floaterSinkerAcceleration,
                "distanceFromCatching": env.distanceFromCatching,
                "bobberInBar": bool(env.bobberInBar),
                "obs": obs.tolist(),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        if done:
            break

    fixture = {
        "seed": seed,
        "fish": {
            "name": env.current_fish["name"],
            "difficulty": int(env.current_fish["difficulty"]),
            "behaviour": env.current_fish["behaviour"],
        },
        "actions": actions[: len(frames) - 1],
        "frames": frames,
    }
    env.close()
    return fixture


def check_rng_selftest() -> None:
    """Sanity: PortableRNG produces a stable known sequence."""
    rng = PortableRNG(42)
    vals = [rng.next_uint32() for _ in range(5)]
    # Locked golden values — update TS test if these change
    expected = [2581720956, 1925393290, 3661312704, 2876485805, 750819978]
    if vals != expected:
        raise AssertionError(f"PortableRNG golden mismatch: {vals} != {expected}")
    print("PortableRNG self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fish", type=str, default="Carp")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    os.chdir(ROOT)
    check_rng_selftest()

    fixture = dump_fixture(seed=args.seed, fish_name=args.fish, steps=args.steps)
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote fixture: {FIXTURE_PATH} ({len(fixture['frames'])} frames)")

    if args.dump_only:
        return 0

    # Run TypeScript parity check
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    cmd = [npx, "--yes", "tsx", TS_CHECK, FIXTURE_PATH]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=os.path.join(ROOT, "competition-website"), shell=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
