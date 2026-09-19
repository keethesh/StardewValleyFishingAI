'use client';

import Link from 'next/link';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import EvolutionShowcase from '@/components/EvolutionShowcase';

export default function EvolutionPage() {
  return (
    <main className="min-h-screen overflow-x-hidden text-[#f7f2ea]">
      <Navbar />
      <div className="relative z-10 mx-auto max-w-6xl px-6 pb-24 pt-28 md:pt-32">
        <header className="mb-12 max-w-3xl space-y-4">
          <p className="text-xs font-bold uppercase tracking-[0.34em] text-[#f7bf47]">
            Reinforcement Learning Journey
          </p>
          <h1 className="font-[family-name:var(--font-pixel)] text-3xl sm:text-4xl md:text-5xl leading-tight text-[#fff7e6] drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
            Watch the AI Learn to Fish
          </h1>
          <p className="text-base sm:text-lg text-[#d8cbba] leading-relaxed">
            Scrub through checkpoints from a single 8-D Dueling Double DQN run. Stage boundaries are empirically calibrated from tap-frequency, duty-cycle, and per-behaviour win rates (not guessed).
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <Link
              href="/"
              className="stardew-btn-wood text-xs px-4 py-2 text-[#fff7e6]"
            >
              ← Back Home
            </Link>
            <Link
              href="/submit"
              className="stardew-btn-gold text-xs px-4 py-2 text-[#24140b]"
            >
              Train Your Own Model
            </Link>
          </div>
        </header>

        <EvolutionShowcase />
      </div>

      <Footer />
    </main>
  );
}
