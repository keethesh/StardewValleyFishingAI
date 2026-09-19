/**
 * Official server-side evaluation.
 *
 * Full catalog (every fish) × N seeds, with a fresh runSeed each submit.
 * Score = Σ_fish mean_seed(caught × difficulty) / Σ difficulty
 * — same formula as training eval_score, but NOT the same seed schedule.
 */

import { FishingGame, type Fish } from './game-logic';
import { ALL_FISH } from './fish-data';
import * as ort from 'onnxruntime-node';

export const OFFICIAL_SEEDS_PER_FISH = 3;
export const OFFICIAL_MAX_STEPS = 2000;

export interface OfficialEpisodeResult {
  fish: string;
  difficulty: number;
  behaviour: string;
  seed: number;
  caught: boolean;
  steps: number;
}

export interface OfficialEvalResult {
  /** Difficulty-weighted catch rate in [0, 1] — leaderboard primary. */
  score: number;
  catchRate: number;
  scoreHard: number;
  nFish: number;
  seedsPerFish: number;
  episodes: number;
  avgSteps: number;
  runSeed: number;
  byBehaviour: Record<string, number>;
  details: OfficialEpisodeResult[];
}

export interface OfficialEvalConfig {
  seedsPerFish?: number;
  maxSteps?: number;
  /** If omitted, a fresh seed is drawn each call. */
  runSeed?: number;
}

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function freshRunSeed(): number {
  return (Date.now() ^ (Math.floor(Math.random() * 0xffffffff))) >>> 0;
}

/** Stable catalog order so every submission faces the same fish list. */
function catalogFish(): Fish[] {
  return [...ALL_FISH].sort((a, b) => {
    const ba = String(a.behaviour).toLowerCase();
    const bb = String(b.behaviour).toLowerCase();
    if (ba !== bb) return ba.localeCompare(bb);
    if (a.difficulty !== b.difficulty) return a.difficulty - b.difficulty;
    return a.name.localeCompare(b.name);
  });
}

async function runEpisode(
  session: ort.InferenceSession,
  fish: Fish,
  seed: number,
  maxSteps: number
): Promise<OfficialEpisodeResult> {
  const game = new FishingGame(fish, seed);
  let steps = 0;

  while (!game.state.done && steps < maxSteps) {
    const state = game.getObservation();
    const tensor = new ort.Tensor('float32', state, [1, 8]);
    const results = await session.run({ state: tensor });
    const q = results.q_values.data as Float32Array;
    const action: 0 | 1 = q[1] > q[0] ? 1 : 0;
    game.step(action);
    steps++;
  }

  return {
    fish: fish.name,
    difficulty: fish.difficulty,
    behaviour: String(fish.behaviour).toLowerCase(),
    seed,
    caught: Boolean(game.state.success),
    steps,
  };
}

/**
 * Official leaderboard eval: every fish × N fresh seeds from runSeed.
 */
export async function runOfficialEvaluation(
  session: ort.InferenceSession,
  config: OfficialEvalConfig = {}
): Promise<OfficialEvalResult> {
  const seedsPerFish = config.seedsPerFish ?? OFFICIAL_SEEDS_PER_FISH;
  const maxSteps = config.maxSteps ?? OFFICIAL_MAX_STEPS;
  const runSeed = config.runSeed ?? freshRunSeed();
  const rng = mulberry32(runSeed);
  const fishList = catalogFish();

  const details: OfficialEpisodeResult[] = [];
  const byBehaviourRaw: Record<string, { attempts: number; successes: number }> = {
    sinker: { attempts: 0, successes: 0 },
    dart: { attempts: 0, successes: 0 },
    smooth: { attempts: 0, successes: 0 },
    mixed: { attempts: 0, successes: 0 },
    floater: { attempts: 0, successes: 0 },
  };

  let weightedSum = 0;
  let weightTotal = 0;
  let hardSum = 0;
  let hardTotal = 0;
  let catches = 0;
  let totalSteps = 0;
  let episodes = 0;

  for (const fish of fishList) {
    let fishWeighted = 0;

    for (let run = 0; run < seedsPerFish; run++) {
      const seed = Math.floor(rng() * 0x7fffffff);
      const result = await runEpisode(session, fish, seed, maxSteps);
      details.push(result);
      episodes++;
      totalSteps += result.steps;

      if (result.caught) {
        fishWeighted += fish.difficulty;
        catches++;
      }

      const b = result.behaviour;
      if (byBehaviourRaw[b]) {
        byBehaviourRaw[b].attempts++;
        if (result.caught) byBehaviourRaw[b].successes++;
      }
    }

    const perFishMean = fishWeighted / seedsPerFish;
    weightedSum += perFishMean;
    weightTotal += fish.difficulty;

    if (fish.difficulty >= 70) {
      hardSum += perFishMean;
      hardTotal += fish.difficulty;
    }
  }

  const byBehaviour: Record<string, number> = {};
  for (const [b, d] of Object.entries(byBehaviourRaw)) {
    byBehaviour[b] = d.attempts ? d.successes / d.attempts : 0;
  }

  return {
    score: weightTotal > 0 ? weightedSum / weightTotal : 0,
    catchRate: episodes > 0 ? catches / episodes : 0,
    scoreHard: hardTotal > 0 ? hardSum / hardTotal : 0,
    nFish: fishList.length,
    seedsPerFish,
    episodes,
    avgSteps: episodes > 0 ? totalSteps / episodes : 0,
    runSeed,
    byBehaviour,
    details,
  };
}
