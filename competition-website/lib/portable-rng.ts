/**
 * Mulberry32 PRNG — bit-identical to portable_rng.py
 * so Python↔TS physics parity tests can share seeds.
 */
export class PortableRNG {
    private state: number;

    constructor(seed: number = 0) {
        this.state = seed >>> 0;
    }

    seed(seed: number): void {
        this.state = seed >>> 0;
    }

    nextUint32(): number {
        this.state = (this.state + 0x6d2b79f5) >>> 0;
        let t = this.state;
        t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
        t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
        return (t ^ (t >>> 14)) >>> 0;
    }

    /** Uniform float in [0, 1). */
    random(): number {
        return this.nextUint32() / 4294967296;
    }

    /** Uniform float in [low, high). */
    uniform(low: number = 0, high: number = 1): number {
        return low + (high - low) * this.random();
    }

    /** Integer in [low, high) — matches numpy.RandomState.randint / PortableRNG.randint. */
    randint(low: number, high: number): number {
        low = Math.trunc(low);
        high = Math.trunc(high);
        if (high <= low) return low;
        const span = high - low;
        return low + Math.floor(this.random() * span);
    }

    choice<T>(seq: T[]): T {
        if (seq.length === 0) throw new Error("Cannot choose from an empty sequence");
        return seq[this.randint(0, seq.length)];
    }
}
