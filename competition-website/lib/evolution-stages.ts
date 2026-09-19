/** Calibrated evolution stages from the 2026-09-19 8-D Dueling DQN run. */

export type BehaviourRates = {
  mixed: number;
  dart: number;
  smooth: number;
  sinker: number;
  floater: number;
};

export type EvolutionStage = {
  id: string;
  name: string;
  episode: number;
  rangeLabel: string;
  insight: string;
  unlocks: string;
  modelUrl: string;
  traceUrl: string;
  tapHz: number;
  centeringError: number;
  rates: BehaviourRates;
};

/**
 * Boundaries recalibrated from checkpoint artefacts.
 * Final published baseline is ep3500 (best eval_score; ep5000 slightly worse).
 */
export const EVOLUTION_STAGES: EvolutionStage[] = [
  {
    id: 'flailing',
    name: 'The Panic Spammer',
    episode: 20,
    rangeLabel: 'Ep 0 – ~200',
    insight:
      'Pins the bar to the ceiling, then dumps it to the floor. Pure exploration — it has not learned to stay off the extremes yet.',
    unlocks: 'Almost nothing (motionless Carp only, rarely).',
    modelUrl: '/models/stages/ep20.onnx',
    traceUrl: '/data/evolution/episode_20.json',
    tapHz: 13.83,
    centeringError: 0.152,
    rates: { mixed: 0, dart: 0, smooth: 0, sinker: 0, floater: 0 },
  },
  {
    id: 'pwm',
    name: 'PWM Hover',
    episode: 500,
    rangeLabel: 'Ep ~200 – 800',
    insight:
      'Thrust is binary; gravity is continuous. Rapid tapping (~27 Hz) lets the bar hover. Centering error collapses and most behaviours unlock.',
    unlocks: 'Smooth / dart / sinker / mixed — floater still shaky.',
    modelUrl: '/models/stages/ep500.onnx',
    traceUrl: '/data/evolution/episode_500.json',
    tapHz: 26.76,
    centeringError: 0.033,
    rates: { mixed: 1, dart: 1, smooth: 1, sinker: 1, floater: 0.75 },
  },
  {
    id: 'cushion',
    name: 'Cushion & Patience',
    episode: 1500,
    rangeLabel: 'Ep ~800 – 2500',
    insight:
      'Soft landings and less overshoot. Floater win rate reaches 100% — the agent stops chasing every bob.',
    unlocks: 'Floaters and hard sinkers consistently.',
    modelUrl: '/models/stages/ep1500.onnx',
    traceUrl: '/data/evolution/episode_1500.json',
    tapHz: 24.32,
    centeringError: 0.026,
    rates: { mixed: 1, dart: 1, smooth: 1, sinker: 1, floater: 1 },
  },
  {
    id: 'baseline',
    name: 'Published Baseline',
    episode: 3500,
    rangeLabel: 'Ep ~2500 – 3500',
    insight:
      'Peak checkpoint from this run (eval_score ≈ 0.97). Still soft on a couple of hard darts — that gap is intentional for the challenge.',
    unlocks: 'Legendaries and nearly the full catalog.',
    modelUrl: '/models/stages/ep3500.onnx',
    traceUrl: '/data/evolution/episode_3500.json',
    tapHz: 22.16,
    centeringError: 0.028,
    rates: { mixed: 1, dart: 1, smooth: 1, sinker: 1, floater: 1 },
  },
];

export const TRAINING_CURVE = [
  { episode: 20, tapHz: 13.83, err: 0.152, overallHint: 0 },
  { episode: 40, tapHz: 28.02, err: 0.172, overallHint: 0.02 },
  { episode: 500, tapHz: 26.76, err: 0.033, overallHint: 0.95 },
  { episode: 1000, tapHz: 24.34, err: 0.037, overallHint: 0.93 },
  { episode: 1500, tapHz: 24.32, err: 0.026, overallHint: 1.0 },
  { episode: 2000, tapHz: 26.72, err: 0.042, overallHint: 0.95 },
  { episode: 2500, tapHz: 23.26, err: 0.029, overallHint: 1.0 },
  { episode: 3000, tapHz: 25.43, err: 0.027, overallHint: 0.98 },
  { episode: 3500, tapHz: 22.16, err: 0.028, overallHint: 1.0 },
  { episode: 4000, tapHz: 24.0, err: 0.029, overallHint: 1.0 },
  { episode: 4500, tapHz: 25.03, err: 0.024, overallHint: 0.98 },
  { episode: 5000, tapHz: 23.8, err: 0.03, overallHint: 0.98 },
];

/** Public demo / published baseline model. */
export const BASELINE_MODEL_URL = '/models/baseline.onnx';
export const BASELINE_EPISODE = 3500;

export function stageNearestToEpisode(episode: number): EvolutionStage {
  let best = EVOLUTION_STAGES[0];
  let bestDist = Math.abs(best.episode - episode);
  for (const s of EVOLUTION_STAGES) {
    const d = Math.abs(s.episode - episode);
    if (d < bestDist) {
      best = s;
      bestDist = d;
    }
  }
  return best;
}
