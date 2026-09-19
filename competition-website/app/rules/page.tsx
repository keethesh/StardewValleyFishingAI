'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

const guideSections = [
    {
        title: '1. What this challenge is',
        accent: 'border-amber-300',
        body: (
            <>
                <p className="text-stone-300">
                    This site is a companion to the video, not a giant standalone platform. The
                    challenge exists for the subset of viewers who immediately want to try training
                    something better than the benchmark model.
                </p>
                <ul className="list-disc space-y-2 pl-5 text-sm leading-7 text-stone-400">
                    <li>You can submit more than once. Your best score is the one that matters.</li>
                    <li>The benchmark starts with the model featured in the video.</li>
                    <li>The real point is generalization, not a clever one-off exploit.</li>
                </ul>
            </>
        ),
    },
    {
        title: '2. Model contract',
        accent: 'border-teal-300',
        body: (
            <>
                <p className="text-stone-300">
                    The submission path is intentionally narrow so the browser demo and the challenge
                    use the same interface.
                </p>
                <ul className="list-disc space-y-2 pl-5 font-mono text-sm leading-7 text-stone-400">
                    <li>Format: `.onnx`</li>
                    <li>Input: `state` with shape `[1, 8]` and `float32` values</li>
                    <li>Output: `q_values` with shape `[1, 2]` and `float32` values</li>
                    <li>Max size: 5 MB</li>
                    <li>Target inference budget: ≤ 16ms per step (60 FPS)</li>
                </ul>
                <div className="mt-3 rounded-2xl border border-white/10 bg-black/20 p-4 font-mono text-xs leading-6 text-stone-400">
                    Observation (8-D): bobber_pos, bobber_vel, bar_pos, bar_vel, bar_height,
                    (bar_center − bobber_pos), in_bar, distanceFromCatching — all normalised.
                </div>
                <p className="text-sm leading-7 text-stone-400">
                    If you use a different training algorithm, that is fine. Just export it so it
                    still satisfies the same contract at inference time.
                </p>
            </>
        ),
    },
    {
        title: '3. How scoring works',
        accent: 'border-emerald-300',
        body: (
            <>
                <p className="text-stone-300">
                    Models are evaluated across a curated spread of fish with different difficulties
                    and movement profiles. Higher difficulty fish are worth more, and consistent
                    catches matter more than a flashy one-off success.
                </p>
                <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-sm leading-7 text-stone-300">
                    Preview score = browser-side evaluation on known seeds
                    <br />
                    Leaderboard score = organizer-side evaluation with hidden seeds and the same model contract
                </div>
                <p className="text-sm leading-7 text-stone-400">
                    In other words: the local preview is useful, but the public board should reward
                    models that actually travel well.
                </p>
            </>
        ),
    },
    {
        title: '4. Guardrails',
        accent: 'border-rose-300',
        body: (
            <>
                <p className="text-stone-300">
                    Change the training code, the network, the loss, the curriculum, the reward
                    shaping, or even the learning algorithm if you want. The guardrails are mostly
                    about keeping the evaluation meaningful.
                </p>
                <ul className="list-disc space-y-2 pl-5 text-sm leading-7 text-stone-400">
                    <li>Do not change the observation size or action size expected by the evaluator.</li>
                    <li>Do not rely on modified physics constants that only exist in your local fork.</li>
                    <li>Do not bake in hard-coded heuristics for specific hidden seeds.</li>
                    <li>Do not depend on helper code that disappears once the model is exported to ONNX.</li>
                </ul>
            </>
        ),
    },
];

export default function RulesPage() {
    return (
        <main className="min-h-screen bg-slate-950 text-stone-100 selection:bg-amber-300 selection:text-slate-950">
            <Navbar />

            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(45,212,191,0.12),_transparent_36%),radial-gradient(circle_at_88%_15%,_rgba(251,191,36,0.12),_transparent_24%),linear-gradient(180deg,_#071118_0%,_#05070a_52%,_#020304_100%)]" />
            </div>

            <div className="relative z-10 mx-auto max-w-5xl px-6 pb-20 pt-32">
                <header className="space-y-6 border-b border-white/10 pb-10">
                    <motion.p
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-xs font-semibold uppercase tracking-[0.32em] text-amber-200/80"
                    >
                        Challenge guide
                    </motion.p>
                    <motion.h1
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="max-w-4xl text-4xl leading-tight text-stone-50 sm:text-5xl"
                    >
                        Everything you need if the demo made you want to train your own.
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.05 }}
                        className="max-w-3xl text-lg leading-8 text-stone-300/85"
                    >
                        The homepage is deliberately lightweight. This page is where the stricter
                        details live: the ONNX contract, how scoring works, and what counts as a fair
                        submission.
                    </motion.p>
                </header>

                <section className="mt-12 space-y-8">
                    {guideSections.map((section, index) => (
                        <motion.article
                            key={section.title}
                            initial={{ opacity: 0, y: 18 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.08 * index }}
                            className="rounded-[28px] border border-white/10 bg-stone-950/50 p-7 shadow-[0_24px_80px_rgba(0,0,0,0.28)]"
                        >
                            <h2 className={`border-l-4 ${section.accent} pl-4 text-2xl font-semibold text-stone-100`}>
                                {section.title}
                            </h2>
                            <div className="mt-5 space-y-4">{section.body}</div>
                        </motion.article>
                    ))}
                </section>

                <section className="mt-12 rounded-[28px] border border-white/10 bg-[linear-gradient(135deg,rgba(251,191,36,0.1),rgba(13,148,136,0.08)_50%,rgba(3,7,18,0.45))] p-8">
                    <div className="space-y-4">
                        <p className="text-xs font-semibold uppercase tracking-[0.28em] text-amber-200/80">
                            Ready to try
                        </p>
                        <h2 className="text-3xl leading-tight text-stone-50">
                            If your model clears the contract, send it through the submission flow.
                        </h2>
                        <p className="max-w-2xl text-base leading-7 text-stone-300/90">
                            You do not need a huge production pipeline here. The bar is simple: train
                            something solid, export it cleanly, and see whether it can outfish the
                            model from the video.
                        </p>
                        <div className="flex flex-wrap gap-3 pt-2">
                            <Link
                                href="/submit"
                                className="rounded-full bg-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5"
                            >
                                Submit a Model
                            </Link>
                            <Link
                                href="/"
                                className="rounded-full border border-white/15 px-6 py-3 text-sm font-semibold text-stone-100 transition-colors hover:border-white/35 hover:bg-white/6"
                            >
                                Back to the Demo
                            </Link>
                        </div>
                    </div>
                </section>
            </div>

            <Footer />
        </main>
    );
}
