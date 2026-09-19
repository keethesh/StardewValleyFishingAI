'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import FishingGameComponent from '@/components/FishingGame';
import { ALL_FISH } from '@/lib/fish-data';
import { BASELINE_MODEL_URL } from '@/lib/evolution-stages';
import type { Fish } from '@/lib/game-logic';
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = '/wasm/';

export default function PlayPage() {
  const playable = useMemo(
    () =>
      ALL_FISH.filter((f) => f.difficulty >= 30).sort(
        (a, b) => a.difficulty - b.difficulty
      ),
    []
  );
  const [fish, setFish] = useState<Fish>(
    playable.find((f) => f.name === 'Catfish') ?? playable[0]
  );
  const [mode, setMode] = useState<'human' | 'ai' | 'race'>('race');
  const [key, setKey] = useState(0);

  const restart = () => setKey((k) => k + 1);

  return (
    <main className="min-h-screen overflow-x-hidden text-[#f7f2ea]">
      <Navbar />
      <div className="relative z-10 mx-auto max-w-6xl px-6 pb-24 pt-28 md:pt-32">
        <header className="mb-10 max-w-3xl space-y-3">
          <p className="text-xs font-bold uppercase tracking-[0.34em] text-[#f7bf47]">
            Interactive Fishing Minigame
          </p>
          <h1 className="font-[family-name:var(--font-pixel)] text-3xl sm:text-4xl text-[#fff7e6] drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
            Human vs AI
          </h1>
          <p className="text-[#d8cbba] leading-relaxed text-sm sm:text-base">
            Step into the pond with exact Stardew Valley physics. Hold left-click or spacebar on desktop (or tap the reel button on mobile) to thrust the green bar. Can you outfish the Episode 3500 baseline?
          </p>
        </header>

        <div className="mb-10 stardew-box p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4">
            <label className="text-xs font-bold uppercase tracking-wider text-[#e6b978] flex items-center gap-2">
              Target Fish:
              <select
                value={fish.name}
                onChange={(e) => {
                  const next = playable.find((f) => f.name === e.target.value);
                  if (next) {
                    setFish(next);
                    restart();
                  }
                }}
                className="rounded border-2 border-[#6b3813] bg-[#180e07] px-3 py-1.5 text-sm font-semibold text-[#fff7e6] focus:outline-none focus:border-[#f7bf47]"
              >
                {playable.map((f) => (
                  <option key={f.name} value={f.name} className="bg-[#180e07] text-[#fff7e6]">
                    {f.name} ({f.difficulty} · {f.behaviour.toUpperCase()})
                  </option>
                ))}
              </select>
            </label>

            <div className="inline-flex rounded-md p-1 bg-[#180e07] border border-[#522a0e]">
              {(
                [
                  ['human', 'Solo Player'],
                  ['ai', 'Solo AI'],
                  ['race', 'Race (Side-by-Side)'],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => {
                    setMode(id);
                    restart();
                  }}
                  className={`px-3 py-1.5 text-xs font-bold rounded transition-all ${
                    mode === id
                      ? 'stardew-btn-gold text-[#24140b]'
                      : 'text-[#d8cbba] hover:text-[#fff7e6]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={restart}
              className="stardew-btn-wood text-xs px-4 py-2 text-[#fff7e6]"
            >
              🔄 Restart Cast
            </button>
            <Link
              href="/evolution"
              className="text-xs font-bold text-[#f7bf47] hover:text-[#ffdb80] underline underline-offset-4"
            >
              Watch it learn →
            </Link>
          </div>
        </div>

        <div className="flex flex-wrap justify-center items-start gap-8 md:gap-12">
          {(mode === 'human' || mode === 'race') && (
            <div className="flex flex-col items-center space-y-3">
              <div className="stardew-slot px-4 py-1 text-center">
                <span className="text-xs font-bold font-[family-name:var(--font-pixel)] text-[#f7bf47] tracking-wider uppercase">
                  Player 1: You
                </span>
              </div>
              <FishingGameComponent key={`h-${key}`} fish={fish} width={200} height={600} />
            </div>
          )}
          {(mode === 'ai' || mode === 'race') && (
            <div className="flex flex-col items-center space-y-3">
              <div className="stardew-slot px-4 py-1 text-center">
                <span className="text-xs font-bold font-[family-name:var(--font-pixel)] text-[#a3e635] tracking-wider uppercase">
                  AI Pilot: Ep 3500
                </span>
              </div>
              <FishingGameComponent
                key={`a-${key}`}
                fish={fish}
                modelUrl={BASELINE_MODEL_URL}
                width={200}
                height={600}
              />
            </div>
          )}
        </div>
      </div>

      <Footer />
    </main>
  );
}
