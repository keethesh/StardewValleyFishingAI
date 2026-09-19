'use client';

import { motion } from 'framer-motion';

const rules = [
    {
        title: "Model Limitations",
        desc: "Models must be exported to ONNX format. Maximum file size is 5MB. Input shape [1, 8], output [1, 2].",
        icon: "📦"
    },
    {
        title: "Inference Speed",
        desc: "The agent must make a decision within 16ms (60fps limit). Slower models will drop frames and likely fail.",
        icon: "⚡"
    },
    {
        title: "Fair Play",
        desc: "No hard-coded logic allowed. We verify submissions by checking the model graph. Generality is key.",
        icon: "🤝"
    },
    {
        title: "Scoring",
        desc: "Official score is difficulty-weighted catch rate on every fish in the catalog (3 fresh seeds each, server-side). Beating the ep3500 baseline is the bar.",
        icon: "🏆"
    }
];

export default function Rules() {
    return (
        <section id="rules" className="py-24 relative">
            {/* Background Elements */}
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-slate-900/50 to-transparent pointer-events-none" />

            <div className="max-w-5xl mx-auto px-6 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    viewport={{ once: true }}
                    className="mb-16 flex flex-col items-center text-center"
                >
                    <div className="w-16 h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent mb-8" />
                    <h2 className="text-3xl md:text-4xl font-bold text-white text-pixel mb-6">
                        Competition <span className="text-purple-400">Rules</span>
                    </h2>
                    <p className="text-slate-400 max-w-xl">
                        To win the prize, your agent must follow these constraints. Breaking them will result in disqualification.
                    </p>
                </motion.div>

                <div className="grid md:grid-cols-2 gap-6">
                    {rules.map((rule, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, scale: 0.95 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.5, delay: idx * 0.1 }}
                            viewport={{ once: true }}
                            className="glass-panel p-6 rounded-xl hover:bg-slate-800/60 transition-colors border-l-4 border-l-cyan-500/0 hover:border-l-cyan-500"
                        >
                            <div className="flex items-start gap-4">
                                <span className="text-3xl bg-slate-800 p-3 rounded-lg">{rule.icon}</span>
                                <div>
                                    <h4 className="font-bold text-white text-lg mb-2">{rule.title}</h4>
                                    <p className="text-slate-400 text-sm leading-relaxed">{rule.desc}</p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="mt-12 text-center"
                >
                    <a href="/rules" className="text-cyan-400 font-bold hover:text-cyan-300 transition-colors border-b border-cyan-400/30 hover:border-cyan-400 pb-0.5">
                        Read Full Regulations & Prize Details →
                    </a>
                </motion.div>
            </div>
        </section>
    );
}
