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
        setStatus(`Stage ready: ${stage.name}`);
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
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
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
                className={`p-3 text-left transition-all rounded ${
                  active
                    ? 'stardew-btn-gold text-[#24140b]'
                    : 'stardew-slot text-[#d8cbba] hover:text-[#fff7e6]'
                }`}
              >
                <div className="font-bold text-xs truncate">{s.name}</div>
                <div className={`text-[10px] font-mono mt-0.5 ${active ? 'text-[#5a330e]' : 'text-[#a89682]'}`}>
                  Episode {s.episode}
                </div>
              </button>
            );
          })}
        </div>

        <div className="stardew-box p-5 space-y-3">
          <p className="text-xs font-bold font-mono uppercase tracking-[0.28em] text-[#f7bf47]">
            {stage.rangeLabel}
          </p>
          <h2 className="font-[family-name:var(--font-pixel)] text-xl sm:text-2xl text-[#fff7e6] drop-shadow">
            {stage.name}
          </h2>
          <p className="text-[#d8cbba] leading-relaxed text-sm">{stage.insight}</p>
          <div className="pt-2 border-t border-[#522a0e] flex items-center gap-2">
            <span className="text-xs font-bold text-[#f7bf47] uppercase font-mono">Unlocks:</span>
            <span className="text-xs text-[#a3e635] font-semibold">{stage.unlocks}</span>
          </div>
        </div>
        <div className="stardew-box p-5 space-y-3">
          <div className="flex items-center justify-between text-xs font-bold font-mono text-[#e6b978]">
            <span className="uppercase tracking-wider">Training Scrubber</span>
            <span className="bg-[#180e07] border border-[#522a0e] px-2 py-0.5 rounded text-[#f7bf47]">
              Episode {scrubEpisode} / {maxEp}
            </span>
          </div>
          <input
            type="range"
            min={20}
            max={maxEp}
            step={20}
            value={scrubEpisode}
            onChange={(e) => setScrubEpisode(Number(e.target.value))}
            className="w-full accent-[#f7bf47] cursor-pointer"
          />
          <div className="h-16 overflow-hidden rounded border-2 border-[#522a0e] bg-[#120a06]">
            <svg viewBox="0 0 500 64" className="h-full w-full" preserveAspectRatio="none">
              <polyline
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
                points={TRAINING_CURVE.map((p) => {
                  const x = (p.episode / maxEp) * 500;
                  const y = 56 - p.tapHz * 1.6;
                  return `${x},${y}`;
                }).join(' ')}
              />
              <polyline
                fill="none"
                stroke="#f7bf47"
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
                stroke="#fff7e6"
                strokeWidth="2"
              />
            </svg>
          </div>
          <div className="flex gap-4 text-xs font-mono">
            <span className="text-[#38bdf8]">● Tap Rate (Hz)</span>
            <span className="text-[#f7bf47]">● Centering Quality</span>
          </div>
        </div>

        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">
            Catch Rate by Fish Behavior
          </p>
          <div className="grid grid-cols-5 gap-2 text-center text-xs">
            {(['sinker', 'dart', 'smooth', 'mixed', 'floater'] as const).map((b) => (
              <div key={b} className="stardew-slot py-2 px-1">
                <div className="uppercase tracking-wider text-[10px] text-[#a89682] font-mono">{b}</div>
                <div className="mt-1 font-bold font-[family-name:var(--font-pixel)] text-xs text-[#fff7e6]">{pct(stage.rates[b])}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 stardew-box p-3">
          <div className="inline-flex rounded p-0.5 bg-[#180e07] border border-[#522a0e] text-xs">
            <button
              type="button"
              onClick={() => {
                setMode('live');
                gameRef.current = new FishingGame(carp);
              }}
              className={`px-3 py-1.5 font-bold rounded transition-all ${mode === 'live' ? 'stardew-btn-gold text-[#24140b]' : 'text-[#d8cbba]'}`}
            >
              Live WASM
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('trace');
                setTraceFrame(0);
              }}
              className={`px-3 py-1.5 font-bold rounded transition-all ${mode === 'trace' ? 'stardew-btn-gold text-[#24140b]' : 'text-[#d8cbba]'}`}
            >
              Trace Replay
            </button>
          </div>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="stardew-btn-wood text-xs px-4 py-1.5 text-[#fff7e6]"
          >
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <span className="text-xs font-mono text-[#a89682]">{status}</span>
        </div>

        <div className="flex flex-col gap-6 sm:flex-row items-center sm:items-start">
          <div className="stardew-box p-2 shadow-2xl">
            <canvas
              ref={canvasRef}
              width={220}
              height={620}
              className="block rounded-sm bg-[#120a06]"
            />
          </div>

          <div className="flex min-w-0 flex-1 flex-col gap-4 w-full">
            <div className="grid grid-cols-2 gap-3">
              <Metric label="Tap Frequency" value={`${telemetry.tapHz.toFixed(0)} Hz`} highlight />
              <Metric
                label="Delta Q (Confidence)"
                value={mode === 'live' ? telemetry.deltaQ.toFixed(3) : 'Replay'}
              />
              <Metric label="Catch Progress" value={`${(telemetry.progress * 100).toFixed(0)}%`} />
              <Metric label="Fish in Bar" value={telemetry.inBar ? 'IN BAR' : 'OUTSIDE'} active={telemetry.inBar} />
            </div>

            <div className="stardew-box p-4 space-y-2">
              <div className="text-xs font-bold uppercase tracking-wider text-[#e6b978] font-mono">
                Instantaneous Thrust History
              </div>
              <div className="flex h-14 items-end gap-px overflow-hidden rounded border border-[#522a0e] bg-[#120a06] p-1">
                {Array.from({ length: TRACE_WINDOW }).map((_, i) => {
                  const v = thrustHistory[i] ?? 0;
                  return (
                    <div
                      key={i}
                      className="w-full transition-all"
                      style={{
                        height: v ? '100%' : '15%',
                        background: v ? '#f7bf47' : '#331d0f',
                      }}
                    />
                  );
                })}
              </div>
              <p className="text-xs font-mono text-[#a89682]">
                Action: <span className="font-bold text-[#fff7e6]">{telemetry.action === 1 ? 'PRESS (THRUST)' : 'RELEASE'}</span> · Stage: ep {stage.episode}
              </p>
            </div>

            {trace && mode === 'trace' && (
              <div className="stardew-slot p-3 text-xs font-mono text-[#d8cbba]">
                Replay: <span className="text-[#f7bf47] font-bold">{trace.trace.fish}</span> (seed {trace.trace.seed}) :{' '}
                <span className={trace.trace.success ? 'text-[#a3e635]' : 'text-[#f87171]'}>{trace.trace.success ? 'CAUGHT' : 'LOST'}</span> ({trace.trace.length} frames)
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, highlight = false, active = false }: { label: string; value: string; highlight?: boolean; active?: boolean }) {
  return (
    <div className="stardew-slot p-3">
      <div className="text-[10px] uppercase font-mono tracking-wider text-[#a89682]">{label}</div>
      <div className={`mt-1 font-[family-name:var(--font-pixel)] text-sm ${
        highlight ? 'text-[#f7bf47]' : active ? 'text-[#a3e635]' : 'text-[#fff7e6]'
      }`}>
        {value}
      </div>
    </div>
  );
}

