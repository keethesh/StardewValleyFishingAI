'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { COMPETITION_CONFIG } from '@/lib/competition-config';

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const videoUrl = `https://www.youtube.com/watch?v=${COMPETITION_CONFIG.youtubeVideoId}`;

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const scrollToSection = (id: string) => {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    return (
        <motion.nav
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5 }}
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'border-b border-white/10 bg-slate-950/80 py-4 backdrop-blur-md' : 'bg-transparent py-6'
                }`}
        >
            <div className="max-w-6xl mx-auto px-6 flex justify-between items-center">
                <div
                    onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                    className="flex items-center gap-2 cursor-pointer group"
                >
                    <span className="text-2xl transition-transform group-hover:scale-110">🎣</span>
                    <span className="text-lg font-semibold tracking-tight text-stone-100">
                        Stardew Fishing AI
                    </span>
                </div>

                <div className="hidden md:flex items-center gap-8 text-sm font-medium text-stone-300">
                    <a href="/play" className="transition-colors hover:text-white">Play</a>
                    <a href="/evolution" className="transition-colors hover:text-white">Evolution</a>
                    <button onClick={() => scrollToSection('challenge')} className="transition-colors hover:text-white">Challenge</button>
                    <button onClick={() => scrollToSection('leaderboard')} className="transition-colors hover:text-white">Leaderboard</button>
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-teal-200 transition-colors hover:text-white"
                    >
                        Watch Video
                    </a>
                </div>

                <a
                    href={COMPETITION_CONFIG.githubRepo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="rounded-full bg-amber-300 px-5 py-2 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5"
                >
                    Starter Repo
                </a>
            </div>
        </motion.nav>
    );
}
