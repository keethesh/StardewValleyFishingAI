'use client';

import { motion } from 'framer-motion';

export default function About() {
    return (
        <section id="about" className="py-20 relative overflow-hidden">
            <div className="max-w-4xl mx-auto px-6 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold mb-6 text-white text-pixel">
                        <span className="text-cyan-400">Technical</span> Details
                    </h2>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
                        Your agent receives raw data from the game memory.
                        It must decide whether to hold the mouse button or release it.
                    </p>
                </motion.div>

                <div className="grid md:grid-cols-2 gap-8">
                    <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        viewport={{ once: true }}
                        className="glass-panel p-8 rounded-2xl relative group overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <span className="text-6xl">👁️</span>
                        </div>
                        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                            <span className="w-1 h-6 bg-cyan-500 rounded-full" />
                            Observation Space
                        </h3>
                        <ul className="space-y-3 text-slate-400 text-sm">
                            <li className="flex justify-between border-b border-white/5 pb-2">
                                <span>Bobber Position</span>
                                <span className="font-mono text-cyan-400">float (0.0 - 1.0)</span>
                            </li>
                            <li className="flex justify-between border-b border-white/5 pb-2">
                                <span>Fish Position</span>
                                <span className="font-mono text-cyan-400">float (0.0 - 1.0)</span>
                            </li>
                            <li className="flex justify-between border-b border-white/5 pb-2">
                                <span>Bobber Velocity</span>
                                <span className="font-mono text-cyan-400">float</span>
                            </li>
                            <li className="flex justify-between border-b border-white/5 pb-2">
                                <span>Perfect Streak</span>
                                <span className="font-mono text-cyan-400">int</span>
                            </li>
                        </ul>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6, delay: 0.4 }}
                        viewport={{ once: true }}
                        className="glass-panel p-8 rounded-2xl relative group overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                            <span className="text-6xl">🎮</span>
                        </div>
                        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                            <span className="w-1 h-6 bg-purple-500 rounded-full" />
                            Action Space
                        </h3>
                        <div className="h-full flex flex-col justify-center">
                            <div className="bg-black/30 p-4 rounded-xl border border-white/5 text-center mb-4">
                                <span className="text-4xl font-bold text-white block mb-1">BINARY</span>
                                <span className="text-xs uppercase tracking-widest text-slate-500">Discrete Action Space</span>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="text-center p-3 bg-slate-800/50 rounded-lg">
                                    <div className="font-mono text-xl text-red-400 mb-1">0</div>
                                    <div className="text-xs text-slate-400">Release</div>
                                </div>
                                <div className="text-center p-3 bg-slate-800/50 rounded-lg">
                                    <div className="font-mono text-xl text-green-400 mb-1">1</div>
                                    <div className="text-xs text-slate-400">Hold</div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
}
