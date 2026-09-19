'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { InferenceSession, Tensor } from 'onnxruntime-web';
import * as ort from 'onnxruntime-web';
import { FishingGame } from '@/lib/game-logic';
import { ALL_FISH } from '@/lib/fish-data';
import {
  EVOLUTION_STAGES,
  TRAINING_CURVE,
  stageNearestToEpisode,
  type EvolutionStage,
} from '@/lib/evolution-stages';

ort.env.wasm.wasmPaths = '/wasm/';

type TraceFrame = {
  t: number;
  action: number;
  bobber_pos: number;
  bar_center: number;
  in_bar: boolean;
};

type TracePayload = {
  episode: number;
  tap_frequency_hz: number;
  mean_centering_error: number;
  trace: {
    fish: string;
    seed: number;
    success: boolean;
    length: number;
    frames: TraceFrame[];
  };
  per_behaviour: { rates: Record<string, number> };
};

type Telemetry = {
  tapHz: number;
  deltaQ: number;
  action: 0 | 1;
  progress: number;
  inBar: boolean;
};

const TRACE_WINDOW = 120;

function pct(n: number) {
  return `${Math.round(n * 100)}%`;
}

export default function EvolutionShowcase() {
  const [stage, setStage] = useState<EvolutionStage>(EVOLUTION_STAGES[1]);
  const [mode, setMode] = useState<'live' | 'trace'>('live');
  const [scrubEpisode, setScrubEpisode] = useState(500);
  const [trace, setTrace] = useState<TracePayload | null>(null);
  const [traceFrame, setTraceFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [status, setStatus] = useState('Loading model…');
  const [telemetry, setTelemetry] = useState<Telemetry>({
    tapHz: 0,
    deltaQ: 0,
    action: 0,
    progress: 0,
    inBar: false,
  });
  const [thrustHistory, setThrustHistory] = useState<number[]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<InferenceSession | null>(null);
  const gameRef = useRef<FishingGame | null>(null);
  const actionWindowRef = useRef<number[]>([]);
  const rafRef = useRef(0);
  const imagesRef = useRef<Record<string, HTMLImageElement>>({});

  const carp = useMemo(
    () => ALL_FISH.find((f) => f.name === 'Carp') ?? ALL_FISH[0],
    []
  );

  // Keep stage in sync with scrubber
  useEffect(() => {
    const nearest = stageNearestToEpisode(scrubEpisode);
    if (nearest.id !== stage.id) {
      setStage(nearest);
    }
  }, [scrubEpisode, stage.id]);

  // Load sprites once
  useEffect(() => {
    const names = [
      'background',
      'catch_bar_top',
      'catch_bar_mid',
      'catch_bar_bot',
      'fish_normal',
      'fish_boss',
    ];
    names.forEach((name) => {
      const img = new Image();
      img.src = `/sprites/${name}.png`;
      img.onload = () => {
        imagesRef.current[name] = img;
      };
    });
  }, []);

  // Load stage artefacts (ONNX + JSON trace)
  useEffect(() => {
    let cancelled = false;
    setStatus('Loading stage…');
    setThrustHistory([]);
    actionWindowRef.current = [];
    setTraceFrame(0);

    (async () => {
      try {
        const [sess, traceRes] = await Promise.all([
          InferenceSession.create(stage.modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          }),
          fetch(stage.traceUrl).then((r) => r.json() as Promise<TracePayload>),
        ]);
        if (cancelled) return;
        sessionRef.current = sess;
        setTrace(traceRes);
        gameRef.current = new FishingGame(carp);
        setStatus(`Stage ready — ${stage.name}`);
        setPlaying(true);
      } catch (err) {
        console.error(err);
        if (!cancelled) setStatus('Failed to load stage artefacts');
      }
    })();

    return () => {
      cancelled = true;
      sessionRef.current = null;
    };
  }, [stage, carp]);

  const drawScene = useCallback(
    (opts: {
      bobberPos: number;
      barTop: number;
      barHeight: number;
      progress: number;
      width: number;
      height: number;
    }) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const { width, height } = opts;
      const images = imagesRef.current;

      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = '#0b1220';
      ctx.fillRect(0, 0, width, height);

      const bg = images['background'];
      if (!bg) return;

      const VISUAL_SCALE = 4;
      const bgW = bg.width * VISUAL_SCALE;
      const bgH = bg.height * VISUAL_SCALE;
      const bgX = (width - bgW) / 2;
      const bgY = (height - bgH) / 2;
      ctx.drawImage(bg, bgX, bgY, bgW, bgH);

      const trackTop = bgY + 25 * (VISUAL_SCALE / 2);
      const trackHeight = bgH - 50 * (VISUAL_SCALE / 2);
      const toCanvas = (val: number) => trackTop + (val / 568) * trackHeight;

      const top = images['catch_bar_top'];
      const mid = images['catch_bar_mid'];
      const bot = images['catch_bar_bot'];
      if (!top || !mid || !bot) return;

      const barY = toCanvas(opts.barTop);
      const barH = (opts.barHeight / 568) * trackHeight;
      const barW = top.width * VISUAL_SCALE;
      const barX = (width - barW) / 2;
      ctx.drawImage(top, barX, barY, barW, top.height * VISUAL_SCALE);
      const midY = barY + top.height * VISUAL_SCALE;
      const botY = barY + barH - bot.height * VISUAL_SCALE;
      const midH = botY - midY;
      if (midH > 0) ctx.drawImage(mid, barX, midY, barW, midH);
      ctx.drawImage(bot, barX, botY, barW, bot.height * VISUAL_SCALE);

      const fishImg = images['fish_normal'];
      if (fishImg) {
        const fishW = fishImg.width * VISUAL_SCALE;
        const fishH = fishImg.height * VISUAL_SCALE;
        ctx.drawImage(
          fishImg,
          (width - fishW) / 2,
          toCanvas(opts.bobberPos) + barH * 0.08,
          fishW,
          fishH
        );
      }

      const progW = 10 * VISUAL_SCALE;
      const progH = bgH - 20 * VISUAL_SCALE;
      const progX = bgX + bgW + 10;
      const progY = bgY + 10 * VISUAL_SCALE;
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(progX, progY, progW, progH);
      const fillH = opts.progress * progH;
      ctx.fillStyle = `rgb(${Math.min(255, Math.floor((2 - 2 * opts.progress) * 255))},${Math.min(255, Math.floor(2 * opts.progress * 255))},0)`;
      ctx.fillRect(progX, progY + progH - fillH, progW, fillH);
    },
    []
  );

  // Main loop
  useEffect(() => {
    const WIDTH = 220;
    const HEIGHT = 620;
    let last = 0;

    const tick = async (ts: number) => {
      rafRef.current = requestAnimationFrame(tick);
      if (!playing) return;
      if (ts - last < 1000 / 60) return;
      last = ts;

      if (mode === 'trace' && trace) {
        const frames = trace.trace.frames;
        const idx = Math.min(traceFrame, frames.length - 1);
        const frame = frames[idx];
        const barHeight = 96; // Carp-ish default visual; trace stores center
        const barTop = frame.bar_center - barHeight / 2;
        drawScene({
          bobberPos: frame.bobber_pos,
          barTop,
          barHeight,
          progress: idx / Math.max(1, frames.length - 1),
          width: WIDTH,
          height: HEIGHT,
        });

        const window = actionWindowRef.current;
        window.push(frame.action);
        if (window.length > 60) window.shift();
        const rises = window.reduce(
          (n, a, i) => (i > 0 && window[i - 1] === 0 && a === 1 ? n + 1 : n),
          0
        );
        setTelemetry({
          tapHz: rises,
          deltaQ: 0,
          action: frame.action as 0 | 1,
          progress: idx / Math.max(1, frames.length - 1),
          inBar: frame.in_bar,
        });
        setThrustHistory((h) => [...h.slice(-(TRACE_WINDOW - 1)), frame.action]);
        setTraceFrame((f) => (f + 1 >= frames.length ? 0 : f + 1));
        return;
      }

      // Live ONNX
      const game = gameRef.current;
      const session = sessionRef.current;
      if (!game || !session) return;

      if (game.state.done) {
        gameRef.current = new FishingGame(carp);
        actionWindowRef.current = [];
        return;
      }

      let action: 0 | 1 = 0;
      let deltaQ = 0;
      try {
        const obs = game.getObservation();
        const tensor = new Tensor('float32', obs, [1, 8]);
        const results = await session.run({ state: tensor });
        const q = results.q_values.data as Float32Array;
        deltaQ = q[1] - q[0];
        action = q[0] > q[1] ? 0 : 1;
      } catch (e) {
        console.error(e);
      }

      const result = game.step(action);
      const window = actionWindowRef.current;
      window.push(action);
      if (window.length > 60) window.shift();
      const rises = window.reduce(
        (n, a, i) => (i > 0 && window[i - 1] === 0 && a === 1 ? n + 1 : n),
        0
      );

      setTelemetry({
        tapHz: rises,
        deltaQ,
        action,
        progress: result.state.distanceFromCatching,
        inBar: result.state.bobberInBar,
      });
      setThrustHistory((h) => [...h.slice(-(TRACE_WINDOW - 1)), action]);

      drawScene({
        bobberPos: result.state.bobberPosition,
        barTop: result.state.bobberBarPos,
        barHeight: result.state.bobberBarHeight,
        progress: result.state.distanceFromCatching,
        width: WIDTH,
        height: HEIGHT,
      });
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, mode, trace, traceFrame, drawScene, carp]);

  const maxEp = TRAINING_CURVE[TRAINING_CURVE.length - 1].episode;

  return (
    <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="space-y-6">
        <div className="flex flex-wrap gap-2">
          {EVOLUTION_STAGES.map((s) => {
            const active = s.id === stage.id;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => {
                  setStage(s);
                  setScrubEpisode(s.episode);
                  setMode('live');
                }}
                className={`rounded-sm border px-3 py-2 text-left text-sm transition ${
                  active
                    ? 'border-amber-300/80 bg-amber-300/15 text-amber-100'
                    : 'border-white/10 bg-white/5 text-stone-300 hover:border-white/25'
                }`}
              >
                <div className="font-medium">{s.name}</div>
                <div className="text-xs text-stone-400">ep {s.episode}</div>
              </button>
            );
          })}
        </div>

        <div className="space-y-3 border-l border-amber-300/40 pl-4">
          <p className="text-xs uppercase tracking-[0.28em] text-amber-200/70">
            {stage.rangeLabel}
          </p>
          <h2 className="font-[family-name:var(--font-pixel)] text-2xl text-stone-50 md:text-3xl">
            {stage.name}
          </h2>
          <p className="max-w-xl text-stone-300 leading-relaxed">{stage.insight}</p>
          <p className="text-sm text-teal-200/80">Unlocks: {stage.unlocks}</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-stone-400">
            <span>Training timeline</span>
            <span>Episode {scrubEpisode}</span>
          </div>
          <input
            type="range"
            min={20}
            max={maxEp}
            step={20}
            value={scrubEpisode}
            onChange={(e) => setScrubEpisode(Number(e.target.value))}
            className="w-full accent-amber-300"
          />
          <div className="grid h-16 grid-cols-[1fr] overflow-hidden rounded-sm border border-white/10 bg-slate-950/80">
            <svg viewBox="0 0 500 64" className="h-full w-full" preserveAspectRatio="none">
              <polyline
                fill="none"
                stroke="rgba(45,212,191,0.85)"
                strokeWidth="2"
                points={TRAINING_CURVE.map((p, i) => {
                  const x = (p.episode / maxEp) * 500;
                  const y = 56 - p.tapHz * 1.6;
                  return `${x},${y}`;
                }).join(' ')}
              />
              <polyline
                fill="none"
                stroke="rgba(251,191,36,0.75)"
                strokeWidth="2"
                points={TRAINING_CURVE.map((p) => {
                  const x = (p.episode / maxEp) * 500;
                  const y = 56 - (1 - Math.min(1, p.err * 6)) * 40;
                  return `${x},${y}`;
                }).join(' ')}
              />
              <line
                x1={(scrubEpisode / maxEp) * 500}
                x2={(scrubEpisode / maxEp) * 500}
                y1="4"
                y2="60"
                stroke="rgba(255,255,255,0.45)"
                strokeWidth="1"
              />
            </svg>
          </div>
          <div className="flex gap-4 text-xs text-stone-500">
            <span className="text-teal-300/80">● tap Hz</span>
            <span className="text-amber-300/80">● centering quality</span>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-2 text-center text-xs">
          {(['sinker', 'dart', 'smooth', 'mixed', 'floater'] as const).map((b) => (
            <div key={b} className="border border-white/10 bg-white/[0.03] px-1 py-2">
              <div className="uppercase tracking-wider text-stone-500">{b}</div>
              <div className="mt-1 text-base text-stone-100">{pct(stage.rates[b])}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="inline-flex rounded-sm border border-white/10 p-0.5 text-sm">
            <button
              type="button"
              onClick={() => {
                setMode('live');
                gameRef.current = new FishingGame(carp);
              }}
              className={`px-3 py-1.5 ${mode === 'live' ? 'bg-teal-400/20 text-teal-100' : 'text-stone-400'}`}
            >
              Live model
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('trace');
                setTraceFrame(0);
              }}
              className={`px-3 py-1.5 ${mode === 'trace' ? 'bg-teal-400/20 text-teal-100' : 'text-stone-400'}`}
            >
              Recorded trace
            </button>
          </div>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded-sm border border-white/15 px-3 py-1.5 text-sm text-stone-200"
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <span className="text-xs text-stone-500">{status}</span>
        </div>

        <div className="flex flex-col gap-4 sm:flex-row">
          <canvas
            ref={canvasRef}
            width={220}
            height={620}
            className="mx-auto border border-white/10 bg-slate-950 shadow-[0_0_40px_rgba(0,0,0,0.45)]"
          />

          <div className="flex min-w-0 flex-1 flex-col gap-4">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <Metric label="Tap frequency" value={`${telemetry.tapHz.toFixed(0)} Hz`} />
              <Metric
                label="ΔQ (press−release)"
                value={mode === 'live' ? telemetry.deltaQ.toFixed(3) : '—'}
              />
              <Metric label="Catch meter" value={`${(telemetry.progress * 100).toFixed(0)}%`} />
              <Metric label="In bar" value={telemetry.inBar ? 'yes' : 'no'} />
            </div>

            <div>
              <div className="mb-1 text-xs uppercase tracking-wider text-stone-500">
                Thrust trace
              </div>
              <div className="flex h-14 items-end gap-px overflow-hidden border border-white/10 bg-slate-950/90 px-1 py-1">
                {Array.from({ length: TRACE_WINDOW }).map((_, i) => {
                  const v = thrustHistory[i] ?? 0;
                  return (
                    <div
                      key={i}
                      className="w-full"
                      style={{
                        height: v ? '100%' : '18%',
                        background: v ? 'rgba(45,212,191,0.85)' : 'rgba(148,163,184,0.25)',
                      }}
                    />
                  );
                })}
              </div>
              <p className="mt-1 text-xs text-stone-500">
                Action {telemetry.action === 1 ? 'PRESS' : 'release'} · checkpoint ep{' '}
                {stage.episode} · tap@eval {stage.tapHz.toFixed(1)} Hz
              </p>
            </div>

            {trace && mode === 'trace' && (
              <p className="text-xs text-stone-400">
                Replay: {trace.trace.fish} seed {trace.trace.seed} —{' '}
                {trace.trace.success ? 'caught' : 'lost'} ({trace.trace.length} frames)
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-white/[0.03] px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-stone-500">{label}</div>
      <div className="mt-1 font-mono text-lg text-stone-100">{value}</div>
    </div>
  );
}
