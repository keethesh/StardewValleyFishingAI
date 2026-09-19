'use client';

import Link from 'next/link';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import EvolutionShowcase from '@/components/EvolutionShowcase';

export default function EvolutionPage() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-slate-950 text-stone-100 selection:bg-amber-300 selection:text-slate-950">
      <Navbar />

      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_0%,_rgba(45,212,191,0.1),_transparent_34%),radial-gradient(circle_at_90%_10%,_rgba(251,191,36,0.1),_transparent_22%),linear-gradient(180deg,_#071118_0%,_#05070a_55%,_#020304_100%)]" />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6 pb-20 pt-28 md:pt-32">
        <header className="mb-14 max-w-2xl space-y-4">
          <p className="text-xs font-semibold uppercase tracking-[0.34em] text-amber-200/80">
            Evolution of the AI
          </p>
          <h1 className="font-[family-name:var(--font-pixel)] text-4xl leading-tight text-stone-50 md:text-5xl">
            Watch it learn to fish
          </h1>
          <p className="text-lg text-stone-300 leading-relaxed">
            Four checkpoints from one 8-D Dueling Double DQN run. Boundaries are calibrated
            from real tap-frequency and per-behaviour win rates — not guessed.
          </p>
          <div className="flex flex-wrap gap-3 pt-2 text-sm">
            <Link
              href="/"
              className="rounded-sm border border-white/15 px-4 py-2 text-stone-300 hover:border-white/30"
            >
              ← Home
            </Link>
            <Link
              href="/submit"
              className="rounded-sm bg-amber-300 px-4 py-2 font-semibold text-slate-950"
            >
              Train your own
            </Link>
          </div>
        </header>

        <EvolutionShowcase />
      </div>

      <Footer />
    </main>
  );
}
