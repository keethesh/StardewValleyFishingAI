"""Summarize checkpoint evolution artefacts for stage-boundary calibration."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "training_logs" / "evolution"


def duty_and_switches(frames: list[dict]) -> tuple[float, float, float]:
    """Return (duty_cycle, switches_per_sec@60fps, mean |centering| proxy)."""
    if not frames:
        return 0.0, 0.0, 0.0
    actions = [int(f["action"]) for f in frames]
    duty = sum(actions) / len(actions)
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    hz = switches / (len(actions) / 60.0)  # half-cycles ≈ tap edges
    # approximate tap Hz as rising edges / second
    rises = sum(1 for i in range(1, len(actions)) if actions[i] == 1 and actions[i - 1] == 0)
    tap_hz = rises / (len(actions) / 60.0)
    return duty, hz, tap_hz


def main() -> None:
    files = sorted(
        EVO.glob("episode_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    print(
        f"{'ep':>6} {'tapHz':>7} {'duty':>6} {'riseHz':>7} {'err':>7} "
        f"{'sink':>6} {'dart':>6} {'smth':>6} {'mix':>6} {'float':>6} {'ok':>4}"
    )
    for path in files:
        d = json.loads(path.read_text())
        ep = d["episode"]
        rates = d["per_behaviour"]["rates"]
        tap = d["tap_frequency_hz"]
        err = d.get("mean_centering_error", float("nan"))
        frames = d.get("trace", {}).get("frames", [])
        duty, edge_hz, rise_hz = duty_and_switches(frames)
        ok = "Y" if d.get("trace", {}).get("success") else "N"

        def pct(k: str) -> str:
            return f"{rates[k]*100:5.0f}%"

        print(
            f"{ep:6d} {tap:7.2f} {duty:6.2f} {rise_hz:7.2f} {err:7.3f} "
            f"{pct('sinker')} {pct('dart')} {pct('smooth')} {pct('mixed')} {pct('floater')} {ok:>4}"
        )


if __name__ == "__main__":
    main()
