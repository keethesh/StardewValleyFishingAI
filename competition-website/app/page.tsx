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
  'The same model featured in the video',
  'Playable in your browser with no setup',
  'Starter repo included if you want to train your own',
];

const QUICK_CONTEXT = [
  {
    label: 'Observation',
    value: '8 numbers',
    description: 'A compact snapshot of the fish, bar, velocity, and catch progress.',
  },
  {
    label: 'Action',
    value: 'Press or release',
    description: 'Every frame the model decides whether to hold the button or let go.',
  },
  {
    label: 'Format',
    value: 'ONNX',
    description: 'The browser demo and challenge both use the same exported model format.',
  },
];

const CHALLENGE_STEPS = [
  'Clone the starter repo and train on the same fishing environment from the video.',
  'Export your model to ONNX so it can run in the browser and in the evaluation pipeline.',
  'Submit it to the challenge and see if you can beat the benchmark model.',
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
    <main className="min-h-screen overflow-x-hidden bg-slate-950 text-stone-100 selection:bg-amber-300 selection:text-slate-950">
      <Navbar />

      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(45,212,191,0.12),_transparent_36%),radial-gradient(circle_at_88%_15%,_rgba(251,191,36,0.12),_transparent_24%),linear-gradient(180deg,_#071118_0%,_#05070a_52%,_#020304_100%)]" />
        <div className="absolute inset-y-0 left-[8%] w-px bg-gradient-to-b from-transparent via-white/8 to-transparent" />
        <div className="absolute inset-y-0 right-[10%] w-px bg-gradient-to-b from-transparent via-white/6 to-transparent" />
      </div>

      <div className="relative z-10 mx-auto flex max-w-6xl flex-col gap-24 px-6 pb-16 pt-28 md:pt-32">
        <section className="grid gap-14 lg:grid-cols-[1.25fr_0.75fr] lg:items-start">
          <div className="space-y-8">
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-xs font-semibold uppercase tracking-[0.34em] text-amber-200/80"
            >
              Companion site for the video
            </motion.p>

            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
              className="max-w-4xl text-4xl leading-[1.08] text-stone-50 sm:text-5xl lg:text-6xl"
            >
              I trained an AI to master Stardew Valley fishing.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.7, ease: 'easeOut' }}
              className="max-w-2xl text-lg leading-8 text-stone-300/85"
            >
              The video tells the story. This page is where you can try the result: watch the model
              fish live, take over the minigame yourself, or grab the starter repo and train one that
              can beat it.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="flex flex-wrap items-center gap-3"
            >
              <button
                onClick={() => jumpToSection('demo')}
                className="rounded-full bg-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5"
              >
                Try the Demo
              </button>
              <button
                onClick={() => jumpToSection('challenge')}
                className="rounded-full border border-white/15 px-6 py-3 text-sm font-semibold text-stone-100 transition-colors hover:border-white/35 hover:bg-white/6"
              >
                Beat the Model
              </button>
              <a
                href={videoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-teal-200 underline decoration-white/15 underline-offset-4 transition-colors hover:text-white"
              >
                Watch the Video
              </a>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="grid gap-4 border-t border-white/10 pt-6 sm:grid-cols-3"
            >
              {HERO_FACTS.map((fact) => (
                <p
                  key={fact}
                  className="max-w-xs text-sm leading-6 text-stone-400 first:text-stone-200"
                >
                  {fact}
                </p>
              ))}
            </motion.div>
          </div>

          <motion.aside
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15, duration: 0.7 }}
            className="relative overflow-hidden rounded-[30px] border border-white/10 bg-stone-950/55 p-8 shadow-[0_30px_80px_rgba(0,0,0,0.35)]"
          >
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-300/60 to-transparent" />
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-teal-200/75">
              Why this page exists
            </p>
            <div className="mt-6 space-y-5">
              <p className="text-2xl leading-tight text-stone-100">
                You watched the build. Now you get to stress test the result.
              </p>
              <p className="text-sm leading-7 text-stone-400">
                Start with the demo to see the model fish in real time. Then switch to manual mode
                and feel how unforgiving the minigame actually is. If you still think you can top
                it, the starter repo and submission flow are waiting below.
              </p>
            </div>

            <div className="mt-8 space-y-4 border-t border-white/8 pt-6 text-sm text-stone-300">
              <div className="flex items-start gap-3">
                <span className="mt-1 h-2 w-2 rounded-full bg-amber-300" />
                <p>Default mode is AI playback so the first thing you see is the model from the video.</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="mt-1 h-2 w-2 rounded-full bg-teal-300" />
                <p>The browser demo and the submission flow both use the same ONNX model contract.</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="mt-1 h-2 w-2 rounded-full bg-stone-300" />
                <p>The challenge is optional. The page still works if you only want to play with the AI.</p>
              </div>
            </div>
          </motion.aside>
        </section>

        <section id="demo" className="scroll-mt-28 space-y-8">
          <div className="flex flex-col gap-4 border-b border-white/10 pb-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-[0.32em] text-amber-200/80">
                Live demo
              </p>
              <h2 className="max-w-3xl text-3xl leading-tight text-stone-50 sm:text-4xl">
                Try the exact model from the video in your browser.
              </h2>
              <p className="max-w-2xl text-base leading-7 text-stone-400">
                Leave AI mode on if you want to watch the trained model handle different fish. Turn
                it off if you want to see why this minigame is such a pain to solve in the first
                place.
              </p>
            </div>
            <a
              href={videoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-teal-200 underline decoration-white/15 underline-offset-4 transition-colors hover:text-white"
            >
              Rewatch the breakdown
            </a>
          </div>

          <div className="grid gap-10 rounded-[34px] border border-white/10 bg-black/20 p-8 shadow-[0_40px_120px_rgba(0,0,0,0.4)] lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
            <div className="order-2 space-y-8 lg:order-1">
              <div className="space-y-5">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.3em] text-stone-500">
                    Fish selector
                  </p>
                  <label className="mt-3 block text-sm text-stone-300">
                    Pick a target species
                  </label>
                  <select
                    className="mt-3 w-full rounded-2xl border border-white/10 bg-stone-950/90 px-4 py-4 text-stone-100 outline-none transition-colors focus:border-amber-300/60"
                    value={selectedFish.name}
                    onChange={(event) => {
                      const fish = ALL_FISH.find((candidate) => candidate.name === event.target.value);
                      if (fish) {
                        setSelectedFish(fish);
                      }
                    }}
                  >
                    {ALL_FISH.map((fish) => (
                      <option key={fish.name} value={fish.name}>
                        {fish.name} - Difficulty {fish.difficulty}
                      </option>
                    ))}
                  </select>
                </div>

                <label className="block rounded-[24px] border border-white/10 bg-stone-950/70 p-5 transition-colors hover:border-white/20">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-sm font-semibold text-stone-100">
                        {useAI ? 'AI Plays' : 'You Play'}
                      </p>
                      <p className="mt-1 text-sm leading-6 text-stone-400">
                        {useAI
                          ? 'This is the trained model from the video running through ONNX Runtime Web.'
                          : 'Manual mode is active. Hold click or tap to control the bar yourself.'}
                      </p>
                    </div>
                    <div
                      className={`relative h-7 w-14 rounded-full transition-colors ${
                        useAI ? 'bg-amber-300' : 'bg-white/12'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={useAI}
                        onChange={(event) => setUseAI(event.target.checked)}
                        className="sr-only"
                      />
                      <div
                        className={`absolute top-1 h-5 w-5 rounded-full bg-slate-950 transition-transform ${
                          useAI ? 'left-8' : 'left-1'
                        }`}
                      />
                    </div>
                  </div>
                </label>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="border border-white/10 bg-stone-950/55 px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-stone-500">
                    Observation
                  </p>
                  <p className="mt-3 text-2xl text-stone-100">14 floats</p>
                  <p className="mt-2 text-sm leading-6 text-stone-400">
                    Fish position, bar state, velocities, difficulty, motion profile, and time.
                  </p>
                </div>
                <div className="border border-white/10 bg-stone-950/55 px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-stone-500">
                    Action
                  </p>
                  <p className="mt-3 text-2xl text-stone-100">Binary</p>
                  <p className="mt-2 text-sm leading-6 text-stone-400">
                    Every frame the model chooses to press or release. Nothing more elaborate than that.
                  </p>
                </div>
              </div>
            </div>

            <div className="order-1 flex justify-center lg:order-2">
              <div className="relative">
                <div className="absolute -inset-5 bg-[radial-gradient(circle,_rgba(251,191,36,0.18),_transparent_55%)] blur-2xl" />
                <div className="relative rounded-[30px] border border-white/12 bg-[#050608] p-4 shadow-2xl">
                  <FishingGameComponent
                    fish={selectedFish}
                    modelUrl={useAI ? '/models/baseline.onnx' : undefined}
                    width={380}
                    height={600}
                  />
                </div>
                {!useAI && (
                  <div className="pointer-events-none absolute -bottom-11 left-1/2 -translate-x-1/2">
                    <span className="rounded-full border border-white/12 bg-stone-950/90 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-stone-300">
                      Hold click or tap
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-start">
          <div className="space-y-6 border-t border-white/10 pt-8">
            <p className="text-xs font-semibold uppercase tracking-[0.32em] text-amber-200/80">
              Quick context
            </p>
            <h2 className="max-w-2xl text-3xl leading-tight text-stone-50">
              Under the hood, the model sees a tiny stream of state and makes one decision every frame.
            </h2>
            <p className="max-w-2xl text-base leading-7 text-stone-400">
              No hand-coded fishing strategy lives in the browser. The exported model gets the current
              state, produces two scores, and the higher one decides whether to press or release.
            </p>
            <div className="grid gap-6 md:grid-cols-3">
              {QUICK_CONTEXT.map((item) => (
                <div key={item.label} className="space-y-3 border-l border-white/10 pl-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.24em] text-stone-500">
                    {item.label}
                  </p>
                  <p className="text-2xl text-stone-100">{item.value}</p>
                  <p className="text-sm leading-6 text-stone-400">{item.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="border border-white/10 bg-stone-950/45 p-7">
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-teal-200/75">
              Companion note
            </p>
            <p className="mt-4 text-lg leading-8 text-stone-200">
              The goal of the page is not to retell the whole project. It is to let you immediately
              test the punchline from the video, then go deeper only if you want to.
            </p>
            <p className="mt-4 text-sm leading-7 text-stone-400">
              That is why the demo comes first, the technical context is short, and the challenge
              sits below as an optional next step instead of the main identity of the site.
            </p>
          </div>
        </section>

        <section
          id="challenge"
          className="scroll-mt-28 rounded-[34px] border border-white/10 bg-[linear-gradient(135deg,rgba(251,191,36,0.1),rgba(13,148,136,0.07)_48%,rgba(3,7,18,0.5))] p-8 shadow-[0_40px_120px_rgba(0,0,0,0.35)]"
        >
          <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-6">
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-[0.32em] text-amber-200/80">
                  Optional challenge
                </p>
                <h2 className="max-w-2xl text-3xl leading-tight text-stone-50 sm:text-4xl">
                  Think you can train something even nastier?
                </h2>
                <p className="max-w-2xl text-base leading-7 text-stone-300/90">
                  If the demo makes you think, &quot;I can beat that,&quot; this is the lane. The
                  challenge is intentionally simple: train on the same environment, export to ONNX,
                  and submit a model that generalizes better than the benchmark from the video.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <a
                  href={COMPETITION_CONFIG.githubRepo}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-full bg-stone-100 px-6 py-3 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5"
                >
                  Get the Starter Repo
                </a>
                <Link
                  href="/submit"
                  className="rounded-full border border-white/15 px-6 py-3 text-sm font-semibold text-stone-100 transition-colors hover:border-white/35 hover:bg-white/6"
                >
                  Submit a Model
                </Link>
                <Link
                  href="/rules"
                  className="rounded-full border border-transparent px-2 py-3 text-sm font-medium text-teal-100 transition-colors hover:text-white"
                >
                  Read the Challenge Guide
                </Link>
              </div>
            </div>

            <div className="space-y-5 border-t border-white/10 pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-stone-500">
                How it works
              </p>
              <ol className="space-y-4">
                {CHALLENGE_STEPS.map((step, index) => (
                  <li key={step} className="flex items-start gap-4">
                    <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/12 bg-black/25 text-sm font-semibold text-stone-100">
                      {index + 1}
                    </span>
                    <p className="text-sm leading-7 text-stone-300">{step}</p>
                  </li>
                ))}
              </ol>
              <p className="text-sm leading-7 text-stone-400">
                The browser preview is instant, but the spirit of the challenge is generalization. The
                guide spells out the model contract, scoring, and guardrails.
              </p>
            </div>
          </div>
        </section>

        <section id="leaderboard" className="scroll-mt-28 space-y-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="space-y-2">
              <p className="text-xs font-semibold uppercase tracking-[0.32em] text-amber-200/80">
                Benchmark board
              </p>
              <h2 className="text-3xl leading-tight text-stone-50 sm:text-4xl">
                Current challenge leaderboard
              </h2>
            </div>
            <p className="max-w-xl text-sm leading-6 text-stone-400">
              The benchmark starts with the model from the video. If someone ships a better one, it
              belongs here.
            </p>
          </div>
          <Leaderboard />
        </section>
      </div>

      <Footer />
    </main>
  );
}
