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
    <main className="min-h-screen overflow-x-hidden bg-slate-950 text-stone-100 selection:bg-amber-300 selection:text-slate-950">
      <Navbar />

      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_0%,_rgba(45,212,191,0.1),_transparent_34%),radial-gradient(circle_at_85%_12%,_rgba(251,191,36,0.1),_transparent_22%),linear-gradient(180deg,_#071118_0%,_#05070a_55%,_#020304_100%)]" />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6 pb-20 pt-28 md:pt-32">
        <header className="mb-10 max-w-2xl space-y-3">
          <p className="text-xs font-semibold uppercase tracking-[0.34em] text-amber-200/80">
            Play
          </p>
          <h1 className="font-[family-name:var(--font-pixel)] text-4xl text-stone-50 md:text-5xl">
            Human vs AI
          </h1>
          <p className="text-stone-300 leading-relaxed">
            Same physics as training. Hold click / tap to thrust. The published baseline
            is episode 3500.
          </p>
        </header>

        <div className="mb-8 flex flex-wrap items-end gap-4">
          <label className="text-sm text-stone-400">
            Fish
            <select
              value={fish.name}
              onChange={(e) => {
                const next = playable.find((f) => f.name === e.target.value);
                if (next) {
                  setFish(next);
                  restart();
                }
              }}
              className="ml-2 rounded-sm border border-white/15 bg-slate-950 px-3 py-2 text-stone-100"
            >
              {playable.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} ({f.difficulty} · {f.behaviour})
                </option>
              ))}
            </select>
          </label>

          <div className="inline-flex rounded-sm border border-white/10 p-0.5 text-sm">
            {(
              [
                ['human', 'You'],
                ['ai', 'AI'],
                ['race', 'Race'],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => {
                  setMode(id);
                  restart();
                }}
                className={`px-3 py-1.5 ${
                  mode === id ? 'bg-teal-400/20 text-teal-100' : 'text-stone-400'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={restart}
            className="rounded-sm border border-white/15 px-3 py-1.5 text-sm text-stone-200"
          >
            Restart
          </button>

          <Link href="/evolution" className="text-sm text-amber-200/80 hover:text-amber-100">
            Watch it learn →
          </Link>
        </div>

        <div
          className={`flex flex-wrap justify-center gap-8 ${
            mode === 'race' ? '' : ''
          }`}
        >
          {(mode === 'human' || mode === 'race') && (
            <div className="space-y-2 text-center">
              <p className="text-xs uppercase tracking-wider text-stone-500">You</p>
              <FishingGameComponent key={`h-${key}`} fish={fish} width={200} height={600} />
            </div>
          )}
          {(mode === 'ai' || mode === 'race') && (
            <div className="space-y-2 text-center">
              <p className="text-xs uppercase tracking-wider text-stone-500">
                Baseline AI
              </p>
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
