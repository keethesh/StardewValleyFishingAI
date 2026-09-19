'use client';

import { FishingGame, Fish } from './game-logic';
import { ALL_FISH } from './fish-data';
import { InferenceSession, Tensor } from 'onnxruntime-web';

/**
 * Browser PREVIEW suite only — fixed fish + fixed seeds.
 * Official leaderboard scoring runs server-side via /api/evaluate
 * (full catalog × 3 fresh seeds). Do not treat this as the contest metric.
 */
export const TEST_FISH: Fish[] = [
    // Easy (difficulty < 40)
    ALL_FISH.find(f => f.name === 'Carp')!,           // 15
    ALL_FISH.find(f => f.name === 'Anchovy')!,        // 30
    ALL_FISH.find(f => f.name === 'Bream')!,          // 35
    ALL_FISH.find(f => f.name === 'Perch')!,          // 35
    ALL_FISH.find(f => f.name === 'Red Snapper')!,    // 40

    // Medium (40-70)
    ALL_FISH.find(f => f.name === 'Rainbow Trout')!,  // 45
    ALL_FISH.find(f => f.name === 'Largemouth Bass')!, // 50
    ALL_FISH.find(f => f.name === 'Tiger Trout')!,    // 60
    ALL_FISH.find(f => f.name === 'Pike')!,           // 60
    ALL_FISH.find(f => f.name === 'Tuna')!,           // 70

    // Hard (70-90)
    ALL_FISH.find(f => f.name === 'Catfish')!,        // 75
    ALL_FISH.find(f => f.name === 'Pufferfish')!,     // 80
    ALL_FISH.find(f => f.name === 'Void Salmon')!,    // 80
    ALL_FISH.find(f => f.name === 'Lingcod')!,        // 85
    ALL_FISH.find(f => f.name === 'Lava Eel')!,       // 90

    // All Legendaries
    ALL_FISH.find(f => f.name === 'Angler')!,         // 85
    ALL_FISH.find(f => f.name === 'Crimsonfish')!,    // 95
    ALL_FISH.find(f => f.name === 'Mutant Carp')!,    // 80
    ALL_FISH.find(f => f.name === 'Glacierfish')!,    // 100
    ALL_FISH.find(f => f.name === 'Legend')!,         // 110
    ALL_FISH.find(f => f.name === 'Ms. Angler')!,     // 85
    ALL_FISH.find(f => f.name === 'Son of Crimsonfish')!, // 95
    ALL_FISH.find(f => f.name === 'Radioactive Carp')!,   // 80
    ALL_FISH.find(f => f.name === 'Glacierfish Jr.')!,    // 100
    ALL_FISH.find(f => f.name === 'Legend II')!,      // 110
].filter(Boolean); // Remove any undefined if fish not found

export interface EpisodeResult {
    fish: Fish;
    seed: number;
    caught: boolean;
    score: number;
}

export interface EvaluationResult {
    totalScore: number;
    maxPossibleScore: number;
    catchRate: number;
    episodes: EpisodeResult[];
}

export type ProgressCallback = (current: number, total: number, fishName: string) => void;

/**
 * Browser preview eval against the fixed public TEST_FISH suite.
 * For leaderboard scores, upload the model to POST /api/evaluate instead.
 */
export async function evaluateModel(
    session: InferenceSession,
    runsPerFish: number = 3,
    onProgress?: ProgressCallback,
    baseSeed: number = 1
): Promise<EvaluationResult> {
    const episodes: EpisodeResult[] = [];
    let totalScore = 0;
    let maxPossibleScore = 0;
    let catches = 0;

    const totalEpisodes = TEST_FISH.length * runsPerFish;
    let currentEpisode = 0;

    for (const fish of TEST_FISH) {
        for (let run = 0; run < runsPerFish; run++) {
            const seed = baseSeed + currentEpisode; // Deterministic seed

            if (onProgress) {
                onProgress(currentEpisode + 1, totalEpisodes, fish.name);
            }

            const result = await runEpisode(session, fish, seed);
            episodes.push(result);

            totalScore += result.score;
            maxPossibleScore += fish.difficulty;
            if (result.caught) catches++;

            currentEpisode++;

            // Small delay to prevent UI freeze
            await new Promise(r => setTimeout(r, 10));
        }
    }

    return {
        totalScore,
        maxPossibleScore,
        catchRate: catches / totalEpisodes,
        episodes
    };
}

/**
 * Run a single episode with the model against a fish.
 */
async function runEpisode(
    session: InferenceSession,
    fish: Fish,
    seed: number
): Promise<EpisodeResult> {
    const game = new FishingGame(fish, seed);

    const MAX_STEPS = 2000;
    const INFERENCE_TIMEOUT_MS = 16;

    for (let step = 0; step < MAX_STEPS && !game.state.done; step++) {
        // Get observation
        const obs = game.getObservation();
        const tensor = new Tensor('float32', obs, [1, 8]);

        let action: 0 | 1 = 0; // Default: release (safe fallback for timeout)

        try {
            // Run inference with timeout protection
            const startTime = performance.now();
            const results = await session.run({ state: tensor });
            const inferenceTime = performance.now() - startTime;

            if (inferenceTime <= INFERENCE_TIMEOUT_MS) {
                const output = results.q_values.data as Float32Array;
                action = output[1] > output[0] ? 1 : 0;
            }
            // If inference too slow, action stays 0 (release penalty)

        } catch (e) {
            // Inference error → default to release
            console.warn('Inference error:', e);
        }

        game.step(action);
    }

    const caught = game.state.success;
    const score = caught ? fish.difficulty : 0;

    return {
        fish,
        seed,
        caught,
        score
    };
}
