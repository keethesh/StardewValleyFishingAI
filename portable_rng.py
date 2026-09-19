"""Portable seeded RNG shared with the TypeScript sim (Mulberry32).

Matches competition-website/lib/portable-rng.ts bit-for-bit so Python↔TS
physics parity tests can use the same seed and action sequence.
"""

from __future__ import annotations


class PortableRNG:
    """Mulberry32 PRNG — identical algorithm in Python and TypeScript."""

    __slots__ = ("_state",)

    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF

    def seed(self, seed: int) -> None:
        self._state = int(seed) & 0xFFFFFFFF

    def next_uint32(self) -> int:
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self._state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def random(self) -> float:
        """Uniform float in [0, 1)."""
        return self.next_uint32() / 4294967296.0

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Uniform float in [low, high)."""
        return low + (high - low) * self.random()

    def randint(self, low: int, high: int) -> int:
        """Integer in [low, high) — matches numpy.RandomState.randint."""
        low = int(low)
        high = int(high)
        if high <= low:
            return low
        span = high - low
        return low + int(self.random() * span)

    def choice(self, seq):
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[self.randint(0, len(seq))]
