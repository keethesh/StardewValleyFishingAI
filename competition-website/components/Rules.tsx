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
            <div className="max-w-5xl mx-auto px-6 relative z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    viewport={{ once: true }}
                    className="mb-14 flex flex-col items-center text-center"
                >
                    <p className="text-xs font-bold uppercase tracking-[0.32em] text-[#f7bf47] mb-3">
                        Official Challenge Regulations
                    </p>
                    <h2 className="text-3xl md:text-4xl font-bold text-[#fff7e6] font-[family-name:var(--font-pixel)] mb-4 drop-shadow">
                        Competition <span className="text-[#f7bf47]">Rules</span>
                    </h2>
                    <p className="text-[#d8cbba] max-w-xl text-sm sm:text-base leading-relaxed">
                        To win the bounty and place on the leaderboard, your model must honor the official 8-D contract.
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
                            className="stardew-box p-6 hover:border-[#d8891d] transition-colors"
                        >
                            <div className="flex items-start gap-4">
                                <span className="text-3xl bg-[#180e07] border border-[#522a0e] p-3 rounded-md">{rule.icon}</span>
                                <div>
                                    <h4 className="font-bold text-[#fff7e6] text-base mb-2 font-[family-name:var(--font-pixel)]">{rule.title}</h4>
                                    <p className="text-[#d8cbba] text-xs sm:text-sm leading-relaxed">{rule.desc}</p>
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
                    <a href="/rules" className="stardew-btn-gold text-xs px-6 py-3 text-[#24140b] inline-block font-bold">
                        Read Full Regulations & Model Contract →
                    </a>
                </motion.div>
            </div>
        </section>
    );
}
