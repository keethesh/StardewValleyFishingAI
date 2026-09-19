'use client';

import Link from 'next/link';
import { useState } from 'react';
import { motion } from 'framer-motion';
import FishingGameComponent from '@/components/FishingGame';
import Leaderboard from '@/components/Leaderboard';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { COMPETITION_CONFIG } from '@/lib/competition-config';
import { ALL_FISH } from '@/lib/fish-data';
import type { Fish } from '@/lib/game-logic';
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = '/wasm/';

const HERO_FACTS = [
  'Exact model featured in the YouTube breakdown',
  'Runs in-browser via WebAssembly with zero setup',
  'Colab starter included to train your own network',
];

const QUICK_CONTEXT = [
  {
    label: 'Observation',
    value: '8 floats',
    description: 'Compact vector: bobber pos/vel, bar pos/vel, bar height, relative distance, in-bar flag, and catch progress.',
  },
  {
    label: 'Action',
    value: 'Binary',
    description: 'Every 16ms frame, the neural net chooses whether to apply upward thrust or release to gravity.',
  },
  {
    label: 'Architecture',
    value: 'Dueling DQN',
    description: 'Separates state value V(s) from action advantage A(s,a) with 3-step temporal-difference returns.',
  },
];

const CHALLENGE_STEPS = [
  'Open the free 1-click Google Colab and train an 8-D Dueling DQN on 4 parallel environments.',
  'Export the checkpoint to a verified .onnx model in one click (~56 KB).',
  'Drop your model into the challenge evaluator to compete for the $10 bounty.',
];

export default function Home() {
  const [selectedFish, setSelectedFish] = useState<Fish>(
    ALL_FISH.find((fish) => fish.name === 'Pufferfish') || ALL_FISH[0]
  );
  const [useAI, setUseAI] = useState(true);

  const videoUrl = `https://www.youtube.com/watch?v=${COMPETITION_CONFIG.youtubeVideoId}`;

  const jumpToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#0a0f16] text-[#f7eedf] selection:bg-[#f6b535] selection:text-[#2a1407]">
      <Navbar />

      {/* Atmospheric Night Fishing Pond Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,_rgba(42,88,118,0.28),_transparent_48%),radial-gradient(circle_at_85%_18%,_rgba(246,181,53,0.1),_transparent_32%),linear-gradient(180deg,_#080d14_0%,_#0e1622_42%,_#070b10_100%)]" />
        <div className="absolute inset-y-0 left-[6%] w-px bg-gradient-to-b from-transparent via-[#e09838]/10 to-transparent" />
        <div className="absolute inset-y-0 right-[6%] w-px bg-gradient-to-b from-transparent via-[#2a5876]/20 to-transparent" />
      </div>

      <div className="relative z-10 mx-auto flex max-w-6xl flex-col gap-20 px-6 pb-20 pt-28 md:pt-32">
        {/* HERO SECTION */}
        <section className="grid gap-12 lg:grid-cols-[1.25fr_0.75fr] lg:items-center">
          <div className="space-y-7">
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="inline-flex items-center gap-2 rounded-full border border-[#f6b535]/30 bg-[#24140b]/80 px-4 py-1.5 backdrop-blur-sm"
            >
              <span className="text-sm">🌟</span>
              <span className="font-pixel text-[11px] font-bold uppercase tracking-wider text-[#f6b535]">
                Companion to the YouTube Breakdown
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
              className="max-w-4xl text-3xl font-extrabold leading-[1.12] text-stone-50 sm:text-5xl lg:text-6xl tracking-tight"
            >
              I trained an AI to master{' '}
              <span className="text-[#f6b535] drop-shadow-[0_2px_10px_rgba(246,181,53,0.3)]">
                Stardew Valley
              </span>{' '}
              fishing.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.6, ease: 'easeOut' }}
              className="max-w-2xl text-base sm:text-lg leading-relaxed text-stone-300"
            >
              Stardew fishing is notoriously brutal. Watch the neural network conquer
              the minigame in real time, take over the controls yourself, or train your
              own model on Google Colab to beat the baseline for the $10 bounty.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="flex flex-wrap items-center gap-4 pt-1"
            >
              <button
                onClick={() => jumpToSection('demo')}
                className="stardew-btn-gold px-6 py-3.5 text-xs font-pixel uppercase tracking-wider"
              >
                Try Live Demo
              </button>
              <Link
                href="/evolution"
                className="stardew-btn-wood px-6 py-3.5 text-xs font-pixel uppercase tracking-wider"
              >
                Watch AI Evolution
              </Link>
              <a
                href={videoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm font-semibold text-amber-200/90 underline decoration-[#f6b535]/40 underline-offset-4 transition-colors hover:text-[#f6b535]"
              >
                <span>▶</span> Watch the Video
              </a>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="grid gap-4 border-t-2 border-[#5c3214]/60 pt-6 sm:grid-cols-3"
            >
              {HERO_FACTS.map((fact) => (
                <div key={fact} className="flex items-start gap-2">
                  <span className="text-amber-400 text-xs mt-0.5">✦</span>
                  <p className="text-xs leading-5 text-stone-300 font-mono">
                    {fact}
                  </p>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Rustic Wood Bulletin Box */}
          <motion.aside
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15, duration: 0.6 }}
            className="stardew-box p-7"
          >
            <div className="flex items-center justify-between border-b border-[#5c3214] pb-3 mb-5">
              <span className="font-pixel text-xs font-bold uppercase tracking-wider text-[#f6b535]">
                Pelican Town Bulletin
              </span>
              <span className="text-xs font-mono text-amber-200/60">Willy&apos;s Shop</span>
            </div>

            <h3 className="font-bold text-lg text-stone-100 mb-3 leading-snug">
              Stress-test the video results right in your browser.
            </h3>
            <p className="text-xs leading-relaxed text-stone-300 mb-6 font-mono">
              Start with the live demo below to inspect the AI pilot on tricky fish like the
              Pufferfish or Catfish. Flip to manual mode anytime to feel the genuine physics.
            </p>

            <div className="space-y-3 border-t border-[#45220c] pt-5">
              <div className="flex items-start gap-3 text-xs text-stone-200 font-mono">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-[#f6b535]" />
                <p>AI Mode runs the official ep3500 baseline model (~56 KB ONNX graph).</p>
              </div>
              <div className="flex items-start gap-3 text-xs text-stone-200 font-mono">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-emerald-400" />
                <p>100% genuine RL: learns PWM tapping, cushion landings, and predictive tracking.</p>
              </div>
              <div className="flex items-start gap-3 text-xs text-stone-200 font-mono">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-amber-200" />
                <p>No downloads required: runs at 60 FPS client-side via WebAssembly.</p>
              </div>
            </div>
          </motion.aside>
        </section>

        {/* LIVE DEMO SECTION */}
        <section id="demo" className="scroll-mt-28 space-y-6">
          <div className="flex flex-col gap-3 border-b-2 border-[#5c3214]/60 pb-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <p className="font-pixel text-xs text-[#f6b535] uppercase tracking-wider">
                Interactive Arena
              </p>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-stone-100 tracking-tight">
                Try the exact model from the video.
              </h2>
              <p className="text-sm text-stone-300 max-w-2xl font-mono">
                Toggle AI Pilot to watch the trained agent respond to erratic movements, or switch to manual control and tap to hold the bar yourself.
              </p>
            </div>
            <Link
              href="/play"
              className="stardew-btn-wood px-4 py-2.5 text-xs font-pixel uppercase tracking-wider self-start lg:self-auto"
            >
              Open Fullscreen Arena →
            </Link>
          </div>

          <div className="stardew-box p-6 md:p-8">
            <div className="grid gap-8 lg:grid-cols-[1fr_auto] lg:items-center">
              {/* Controls Column */}
              <div className="space-y-6 order-2 lg:order-1">
                <div className="space-y-4">
                  <div>
                    <label className="block font-pixel text-xs text-amber-200/80 uppercase tracking-wider mb-2">
                      Select Fish Species
                    </label>
                    <select
                      className="w-full stardew-slot px-4 py-3 text-sm text-stone-100 outline-none border-2 border-[#5c3214] focus:border-[#f6b535] cursor-pointer font-mono"
                      value={selectedFish.name}
                      onChange={(event) => {
                        const fish = ALL_FISH.find((candidate) => candidate.name === event.target.value);
                        if (fish) {
                          setSelectedFish(fish);
                        }
                      }}
                    >
                      {ALL_FISH.map((fish) => (
                        <option key={fish.name} value={fish.name} className="bg-[#1a0e07] text-stone-100">
                          {fish.name} (Difficulty {fish.difficulty}, {fish.behaviour})
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Mode Toggle */}
                  <div className="stardew-slot p-4 border border-[#5c3214]/80">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="font-pixel text-xs uppercase text-[#f6b535]">
                          {useAI ? 'Mode: AI Pilot' : 'Mode: Manual Angler'}
                        </p>
                        <p className="mt-1 text-xs text-stone-300 font-mono leading-relaxed">
                          {useAI
                            ? 'Neural network evaluates state every 16ms and decides button pressure.'
                            : 'Manual mode active: hold mouse click, spacebar, or screen tap to reel.'}
                        </p>
                      </div>
                      <button
                        onClick={() => setUseAI(!useAI)}
                        className={`px-4 py-2 text-xs font-pixel uppercase rounded border transition-colors cursor-pointer ${
                          useAI
                            ? 'bg-[#f6b535] text-[#24140b] font-bold border-[#8a4e0a]'
                            : 'bg-[#3d200d] text-amber-200 border-[#6b3813] hover:bg-[#522c12]'
                        }`}
                      >
                        {useAI ? 'AI ON' : 'MANUAL'}
                      </button>
                    </div>
                  </div>
                </div>

                {/* 8-D Contract Telemetry Badges */}
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="stardew-slot p-4">
                    <p className="font-pixel text-[10px] text-amber-200/70 uppercase">
                      Input Contract
                    </p>
                    <p className="mt-1 font-pixel text-lg text-[#f6b535]">8 Floats</p>
                    <p className="mt-1 text-xs text-stone-300 font-mono leading-5">
                      Fish kinematics, bar position, velocity, error, and catch gauge.
                    </p>
                  </div>
                  <div className="stardew-slot p-4">
                    <p className="font-pixel text-[10px] text-amber-200/70 uppercase">
                      Decision Frequency
                    </p>
                    <p className="mt-1 font-pixel text-lg text-[#f6b535]">60 FPS</p>
                    <p className="mt-1 text-xs text-stone-300 font-mono leading-5">
                      Binary Q-value comparison (Press vs. Release) under 1ms latency.
                    </p>
                  </div>
                </div>
              </div>

              {/* Game Canvas Column */}
              <div className="order-1 lg:order-2 flex flex-col items-center justify-center">
                <div className="relative p-3 bg-[#120904] border-4 border-[#6b3813] rounded-lg shadow-2xl">
                  <FishingGameComponent
                    fish={selectedFish}
                    modelUrl={useAI ? '/models/baseline.onnx' : undefined}
                    width={320}
                    height={520}
                  />
                  {!useAI && (
                    <div className="mt-2 text-center">
                      <span className="font-pixel text-[10px] uppercase tracking-wider text-[#f6b535] bg-[#24140b] px-3 py-1 rounded border border-[#5c3214]">
                        Click canvas or tap to reel
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* QUICK CONTEXT SECTION */}
        <section className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-start">
          <div className="space-y-5 border-t-2 border-[#5c3214]/60 pt-6">
            <p className="font-pixel text-xs text-[#f6b535] uppercase tracking-wider">
              Under the Hood
            </p>
            <h2 className="text-2xl sm:text-3xl font-bold text-stone-100 tracking-tight">
              A lean network solving 1-D continuous physics.
            </h2>
            <p className="text-sm text-stone-300 leading-relaxed font-mono">
              The agent does not use cheat codes or hardcoded timers. It observes Newtonian
              mechanics (gravity, velocity, bar bounce damping) and learned through trial and error
              that rhythmically tapping (PWM hover) beats frantic holding.
            </p>
            <div className="grid gap-4 sm:grid-cols-3 pt-2">
              {QUICK_CONTEXT.map((item) => (
                <div key={item.label} className="stardew-slot p-4 space-y-1.5">
                  <p className="font-pixel text-[10px] text-amber-200/60 uppercase">
                    {item.label}
                  </p>
                  <p className="font-pixel text-base text-[#f6b535]">{item.value}</p>
                  <p className="text-xs text-stone-300 font-mono leading-5">{item.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="stardew-box p-6 space-y-3">
            <div className="flex items-center gap-2 border-b border-[#5c3214] pb-2">
              <span className="text-base">📜</span>
              <p className="font-pixel text-xs text-[#f6b535] uppercase">
                Architecture Spec
              </p>
            </div>
            <p className="text-sm font-semibold text-stone-100">
              Double DQN with Dueling Advantage & N-Step Returns
            </p>
            <p className="text-xs text-stone-300 leading-relaxed font-mono">
              By separating the value of state V(s) from action advantage A(s,a), the network
              knows when it is in a safe hover versus when an immediate thrust is required.
              The model compiles to a lightweight 56 KB ONNX graph that runs instantly in any browser.
            </p>
            <div className="pt-2">
              <Link
                href="/evolution"
                className="inline-flex items-center gap-1.5 text-xs font-pixel text-amber-300 hover:text-amber-200 underline underline-offset-4"
              >
                Inspect the 4 learning stages →
              </Link>
            </div>
          </div>
        </section>

        {/* CHALLENGE / $10 BOUNTY SECTION */}
        <section
          id="challenge"
          className="scroll-mt-28 stardew-box p-8 border-2 border-[#8b5523]"
        >
          <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
            <div className="space-y-6">
              <div className="space-y-2.5">
                <div className="inline-flex items-center gap-2 rounded bg-[#331a0b] px-3 py-1 border border-[#6b3813]">
                  <span className="text-xs">🏆</span>
                  <span className="font-pixel text-[10px] font-bold text-[#f6b535] uppercase">
                    Community Challenge
                  </span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-stone-100 tracking-tight">
                  Can your model beat the baseline?
                </h2>
                <p className="text-sm text-stone-300 leading-relaxed font-mono">
                  Think your reinforcement learning pipeline can outperform ep3500 on legendary fish?
                  Train your own model using the free Colab notebook, export to ONNX, and submit it.
                  The highest score across hidden test seeds when the timer expires wins the $10 prize.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <a
                  href={COMPETITION_CONFIG.githubRepo}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="stardew-btn-gold px-5 py-3 text-xs font-pixel uppercase tracking-wider"
                >
                  Starter Colab & Repo
                </a>
                <Link
                  href="/submit"
                  className="stardew-btn-wood px-5 py-3 text-xs font-pixel uppercase tracking-wider"
                >
                  Submit Model (.onnx)
                </Link>
                <Link
                  href="/rules"
                  className="text-xs font-pixel text-amber-300/80 hover:text-[#f6b535] underline underline-offset-4 ml-2"
                >
                  Full Regulations
                </Link>
              </div>
            </div>

            <div className="space-y-4 border-t-2 border-[#5c3214]/60 pt-6 lg:border-l-2 lg:border-t-0 lg:pl-8 lg:pt-0">
              <p className="font-pixel text-xs text-amber-200/70 uppercase tracking-wider">
                How to Enter
              </p>
              <ol className="space-y-3 font-mono text-xs">
                {CHALLENGE_STEPS.map((step, index) => (
                  <li key={step} className="flex items-start gap-3">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-[#452410] border border-[#6b3813] font-pixel text-[10px] font-bold text-[#f6b535]">
                      {index + 1}
                    </span>
                    <p className="leading-5 text-stone-300 pt-0.5">{step}</p>
                  </li>
                ))}
              </ol>
              <p className="text-[11px] text-amber-200/50 font-mono border-t border-[#452410] pt-3">
                Client preview checks on known seeds; official scores are verified server-side on hidden test seeds.
              </p>
            </div>
          </div>
        </section>

        {/* LEADERBOARD SECTION */}
        <section id="leaderboard" className="scroll-mt-28 space-y-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between border-b-2 border-[#5c3214]/60 pb-4">
            <div>
              <p className="font-pixel text-xs text-[#f6b535] uppercase tracking-wider">
                Leaderboard
              </p>
              <h2 className="text-2xl sm:text-3xl font-bold text-stone-100 tracking-tight">
                Current Challenge Standings
              </h2>
            </div>
            <p className="text-xs text-amber-200/60 font-mono">
              Live evaluations across all 25 difficulty tiers
            </p>
          </div>
          <Leaderboard />
        </section>
      </div>

      <Footer />
    </main>
  );
}
