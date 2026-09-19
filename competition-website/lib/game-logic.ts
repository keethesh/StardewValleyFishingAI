/**
 * Fishing minigame physics — line-by-line port of environment.py
 * (_update_game_logic, _calculate_reward, _get_observation).
 *
 * Uses PortableRNG (Mulberry32) identical to portable_rng.py so a shared
 * seed + action sequence produces matching positions in Python and TS.
 */

import { PortableRNG } from './portable-rng';

export const OBS_DIM = 8;

export interface Fish {
    name: string;
    difficulty: number;
    behaviour: 'mixed' | 'dart' | 'smooth' | 'sinker' | 'floater';
    legendary_tier?: string;
}

export type Action = 0 | 1; // 0 = None, 1 = Press

export interface GameState {
    bobberPosition: number;
    bobberSpeed: number;
    bobberAcceleration: number;
    bobberTargetPosition: number;

    bobberBarPos: number;
    bobberBarSpeed: number;
    bobberBarHeight: number;

    distanceFromCatching: number;
    bobberInBar: boolean;

    floaterSinkerAcceleration: number;
    fishSize: number;
    fishSizeReductionTimer: number;
    whichBobber: number;
    beginnersRod: boolean;
    handledFishResult: boolean;

    done: boolean;
    success: boolean;
    currentTimestep: number;
    maxTimesteps: number;
}

export const PHYSICS = {
    TRACK_HEIGHT: 568,
    POS_MIN: 0,
    POS_MAX: 532,
    BAR_MAX: 568,
    GRAVITY: 0.25,
    LIFT: -0.25,
    BAR_HEIGHT_DEFAULT: 96,
    FPS: 60,
    TIMESTEP: 16,
};

const BEHAVIOR_TYPES: Record<string, number> = {
    mixed: 0,
    dart: 1,
    smooth: 2,
    sinker: 3,
    floater: 4,
};

export class FishingGame {
    state: GameState;
    fish: Fish;
    rng: PortableRNG;
    private readonly augmentFish: boolean;

    private readonly normConstants = {
        height: 568,
        maxFishSize: 20,
        speedNorm: 10.0,
        maxTimesteps: 2000,
    };

    constructor(fish: Fish, seed: number = 0, opts?: { augmentFish?: boolean }) {
        this.fish = { ...fish };
        this.rng = new PortableRNG(seed);
        this.augmentFish = opts?.augmentFish ?? false;
        if (this.augmentFish) {
            const variation = this.rng.uniform(-0.1, 0.1);
            this.fish.difficulty = Math.max(
                1,
                Math.min(110, Math.trunc(this.fish.difficulty * (1.0 + variation)))
            );
        }
        this.state = this.getInitialState();
    }

    private getInitialState(): GameState {
        return {
            bobberPosition: 100.0,
            bobberSpeed: 0.0,
            bobberAcceleration: 0.0,
            bobberTargetPosition: 200.0,

            bobberBarPos: 200.0,
            bobberBarSpeed: 0.0,
            bobberBarHeight: 96,

            distanceFromCatching: 0.5,
            bobberInBar: false,

            floaterSinkerAcceleration: 0.0,
            fishSize: 10,
            fishSizeReductionTimer: 800,
            whichBobber: 0,
            beginnersRod: false,
            handledFishResult: false,

            done: false,
            success: false,
            currentTimestep: 0,
            maxTimesteps: 2000,
        };
    }

    reset(seed?: number): Float32Array {
        if (seed !== undefined) {
            this.rng.seed(seed);
            if (this.augmentFish) {
                // Re-roll difficulty from original fish on reseed
                // Callers that need exact parity should leave augmentFish false.
            }
        }
        this.state = this.getInitialState();
        return this.getObservation();
    }

    step(action: Action): { state: GameState; reward: number; done: boolean; obs: Float32Array } {
        if (this.state.done) {
            return { state: this.state, reward: 0, done: true, obs: this.getObservation() };
        }

        const prevDistance = this.state.distanceFromCatching;
        const buttonPressed = action === 1;

        this.updateGameLogic(16, buttonPressed);
        this.state.currentTimestep += 1;

        // Match Python: reward is computed before the timeout latch below.
        const reward = this.calculateReward(prevDistance);

        let done =
            this.state.handledFishResult ||
            this.state.currentTimestep >= this.state.maxTimesteps;

        if (this.state.currentTimestep >= this.state.maxTimesteps && !this.state.handledFishResult) {
            this.state.handledFishResult = true;
            this.state.distanceFromCatching = 0.0;
            done = true;
        }

        this.state.done = done;
        this.state.success = done && this.state.distanceFromCatching >= 1.0;

        return { state: this.state, reward, done, obs: this.getObservation() };
    }

    /**
     * Exact port of FishingMinigameEnv._update_game_logic
     */
    private updateGameLogic(timeElapsed: number, buttonPressed: boolean): void {
        const rng = this.rng;
        const diff = this.fish.difficulty;
        const motion = BEHAVIOR_TYPES[this.fish.behaviour] ?? 0;
        let pos = this.state.bobberPosition;
        let target = this.state.bobberTargetPosition;

        // Attempt to set a new target occasionally
        if (
            rng.random() < (diff * (motion === 2 ? 20.0 : 1.0)) / 4000.0 &&
            (motion !== 2 || target === -1.0)
        ) {
            const num1 = 548.0 - pos;
            const num2 = Math.min(99.0, diff + rng.randint(10, 45)) * 0.01;
            const lo = Math.trunc(Math.max(-pos, -num1));
            const hi = Math.trunc(num1);
            target = pos + rng.randint(lo, hi) * num2;
        }

        // Floater/sinker adjustments
        let fa = this.state.floaterSinkerAcceleration;
        if (motion === 4) {
            fa = Math.max(fa - 0.01, -1.5);
        } else if (motion === 3) {
            fa = Math.min(fa + 0.01, 1.5);
        }
        this.state.floaterSinkerAcceleration = fa;

        // Move bobber towards target
        if (Math.abs(pos - target) > 3.0 && target !== -1.0) {
            const bobberAcc =
                (target - pos) / (rng.randint(10, 30) + (100.0 - Math.min(100.0, diff)));
            this.state.bobberAcceleration = bobberAcc;
            this.state.bobberSpeed += (bobberAcc - this.state.bobberSpeed) / 5.0;
        } else {
            if (motion === 2 || rng.random() >= diff / 2000.0) {
                target = -1.0;
            } else {
                target =
                    pos +
                    (rng.random() < 0.5 ? rng.randint(-100, -51) : rng.randint(50, 101));
            }
        }

        if (motion === 1 && rng.random() < diff / 1000.0) {
            const spread = Math.trunc(diff) * 2;
            target =
                pos +
                (rng.random() < 0.5
                    ? rng.randint(-100 - spread, -51)
                    : rng.randint(50, 101 + spread));
        }

        // Clamp target
        this.state.bobberTargetPosition = Math.max(-1.0, Math.min(target, 548.0));

        // Update bobber position
        pos = pos + this.state.bobberSpeed + fa;
        this.state.bobberPosition = Math.max(0.0, Math.min(pos, 532.0));

        // Check if bobber in bar
        this.state.bobberInBar =
            this.state.bobberPosition >= this.state.bobberBarPos &&
            this.state.bobberPosition <= this.state.bobberBarPos + this.state.bobberBarHeight;

        // Move the bobber bar based on input
        let num4 = buttonPressed ? -0.25 : 0.25;
        if (
            buttonPressed &&
            num4 < 0.0 &&
            (this.state.bobberBarPos === 0.0 ||
                this.state.bobberBarPos === 568 - this.state.bobberBarHeight)
        ) {
            this.state.bobberBarSpeed = 0.0;
        }

        if (this.state.bobberInBar) {
            num4 *= this.state.whichBobber === 691 ? 0.3 : 0.6;
            if (this.state.whichBobber === 691) {
                const midPoint = this.state.bobberBarPos + this.state.bobberBarHeight / 2.0;
                if (this.state.bobberPosition < midPoint) {
                    this.state.bobberBarSpeed -= 0.2;
                } else {
                    this.state.bobberBarSpeed += 0.2;
                }
            }
        }

        this.state.bobberBarSpeed += num4;
        this.state.bobberBarPos += this.state.bobberBarSpeed;

        // Constrain the bar
        if (this.state.bobberBarPos + this.state.bobberBarHeight > 568.0) {
            this.state.bobberBarPos = 568.0 - this.state.bobberBarHeight;
            this.state.bobberBarSpeed = -(
                (this.state.bobberBarSpeed * 2.0) /
                3.0 *
                (this.state.whichBobber === 692 ? 0.1 : 1.0)
            );
        } else if (this.state.bobberBarPos < 0.0) {
            this.state.bobberBarPos = 0.0;
            this.state.bobberBarSpeed = -((this.state.bobberBarSpeed * 2.0) / 3.0);
        }

        // Update distance from catching
        if (this.state.bobberInBar) {
            this.state.distanceFromCatching += 1.0 / 500.0;
        } else {
            this.state.fishSizeReductionTimer -= timeElapsed;
            if (this.state.fishSizeReductionTimer <= 0) {
                this.state.fishSize = Math.max(5, this.state.fishSize - 1);
                this.state.fishSizeReductionTimer = 800;
            }
            this.state.distanceFromCatching -=
                this.state.whichBobber === 694 || this.state.beginnersRod
                    ? 1.0 / 500.0
                    : 3.0 / 1000.0;
        }

        this.state.distanceFromCatching = Math.max(
            0.0,
            Math.min(1.0, this.state.distanceFromCatching)
        );

        if (this.state.distanceFromCatching <= 0.0 || this.state.distanceFromCatching >= 1.0) {
            this.state.handledFishResult = true;
        }
    }

    /**
     * Exact port of FishingMinigameEnv._calculate_reward
     */
    calculateReward(prevDistance: number): number {
        const s = this.state;
        const bsp = s.bobberSpeed;
        const bbp = s.bobberBarPos;
        const bbh = s.bobberBarHeight;
        const dfc = s.distanceFromCatching;
        const barCenter = bbp + bbh * 0.5;

        const progressRew = (dfc - prevDistance) * 20.0;

        let inBar: number;
        if (s.bobberInBar) {
            inBar = 0.02;
            const fpib = (s.bobberPosition - barCenter) / (bbh * 0.5);
            inBar += 0.05 * Math.exp(-4.0 * fpib * fpib);
            if (this.fish.behaviour.toLowerCase() === 'floater') {
                const bsm = Math.abs(bsp / this.normConstants.speedNorm);
                if (bsm < 0.05 && Math.abs(fpib) < 0.3) {
                    inBar += 0.1;
                }
                if (bsm > 0.3) {
                    inBar -= 0.03 * bsm;
                }
            }
        } else {
            inBar = -0.05;
        }

        const distToFish = Math.abs(barCenter - s.bobberPosition);
        const proximity = -0.025 * (distToFish / PHYSICS.TRACK_HEIGHT);
        const velPenalty = -0.008 * Math.abs(bsp - s.bobberBarSpeed);
        const movePenalty = -0.005 * Math.abs(s.bobberBarSpeed);
        const early = dfc < 0.3 && s.bobberInBar ? 0.05 : 0.0;

        const diffFac = this.fish.difficulty * 0.02;

        if (s.handledFishResult) {
            if (dfc >= 1.0) {
                const tb = 15.0 * (1.0 - s.currentTimestep / s.maxTimesteps);
                return 50.0 * diffFac + (s.fishSize / 20.0) * 20.0 + tb;
            }
            return -10.0 - 15.0 * (s.currentTimestep / s.maxTimesteps);
        }

        return (
            (progressRew + inBar + proximity + velPenalty + movePenalty + early - 0.01) *
            diffFac
        );
    }

    /**
     * 8-D observation — matches environment._get_observation
     */
    getObservation(): Float32Array {
        const s = this.state;
        const nc = this.normConstants;
        const barCenter = s.bobberBarPos + s.bobberBarHeight * 0.5;

        return new Float32Array([
            s.bobberPosition / nc.height,
            s.bobberSpeed / nc.speedNorm,
            s.bobberBarPos / nc.height,
            s.bobberBarSpeed / nc.speedNorm,
            s.bobberBarHeight / nc.height,
            (barCenter - s.bobberPosition) / nc.height,
            s.bobberInBar ? 1.0 : 0.0,
            s.distanceFromCatching,
        ]);
    }
}
