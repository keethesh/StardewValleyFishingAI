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
                <div className="mt-3 rounded border-2 border-[#522a0e] bg-[#120a06] p-4 font-mono text-xs leading-6 text-[#e6b978]">
                    Observation (8-D): bobber_pos, bobber_vel, bar_pos, bar_vel, bar_height,
                    (bar_center - bobber_pos), in_bar, distanceFromCatching (all normalised).
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
        <main className="min-h-screen text-[#f7f2ea]">
            <Navbar />
            <div className="relative z-10 mx-auto max-w-5xl px-6 pb-24 pt-28 md:pt-32">
                <header className="space-y-4 border-b border-[#522a0e] pb-10">
                    <motion.p
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-xs font-bold uppercase tracking-[0.32em] text-[#f7bf47]"
                    >
                        Pelican Town Challenge Guide
                    </motion.p>
                    <motion.h1
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="max-w-4xl font-[family-name:var(--font-pixel)] text-3xl sm:text-4xl text-[#fff7e6] leading-tight drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]"
                    >
                        Official Regulations & Model Contract
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.05 }}
                        className="max-w-3xl text-sm sm:text-base leading-relaxed text-[#d8cbba]"
                    >
                        Everything you need to train your own fishing agent: the 8-D ONNX interface, how official server scoring works, and the fair-play guardrails to claim the bounty.
                    </motion.p>
                </header>

                <section className="mt-12 space-y-8">
                    {guideSections.map((section, index) => (
                        <motion.article
                            key={section.title}
                            initial={{ opacity: 0, y: 18 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.08 * index }}
                            className="stardew-box p-6 sm:p-8"
                        >
                            <h2 className="font-[family-name:var(--font-pixel)] text-xl sm:text-2xl text-[#f7bf47] mb-4 drop-shadow">
                                {section.title}
                            </h2>
                            <div className="space-y-4 text-sm leading-relaxed text-[#d8cbba]">{section.body}</div>
                        </motion.article>
                    ))}
                </section>

                <section className="mt-12 stardew-box p-8 bg-gradient-to-r from-[#2a170e] via-[#331c11] to-[#24140b]">
                    <div className="space-y-4">
                        <p className="text-xs font-bold uppercase tracking-[0.28em] text-[#f7bf47] font-mono">
                            Ready to Take the Challenge?
                        </p>
                        <h2 className="font-[family-name:var(--font-pixel)] text-2xl sm:text-3xl text-[#fff7e6] leading-tight">
                            Train your agent and test it against the community.
                        </h2>
                        <p className="max-w-2xl text-sm leading-relaxed text-[#d8cbba]">
                            You do not need a cluster of GPUs. Train for 15 minutes in our free Google Colab, export the 56 KB model, and see if your agent can outfish Episode 3500.
                        </p>
                        <div className="flex flex-wrap gap-4 pt-3">
                            <Link
                                href="/submit"
                                className="stardew-btn-gold text-xs px-6 py-3 text-[#24140b] font-bold"
                            >
                                Submit Your Model (.onnx)
                            </Link>
                            <Link
                                href="/"
                                className="stardew-btn-wood text-xs px-6 py-3 text-[#fff7e6]"
                            >
                                ← Back to Companion Demo
                            </Link>
                        </div>
                    </div>
                </section>
            </div>

            <Footer />
        </main>
    );
}
