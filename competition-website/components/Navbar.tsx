'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import Link from 'next/link';
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
            transition={{ duration: 0.4 }}
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-200 ${
                scrolled 
                    ? 'border-b-2 border-[#6b3813] bg-[#140a05]/95 py-3 shadow-[0_6px_20px_rgba(0,0,0,0.7)] backdrop-blur-md' 
                    : 'bg-gradient-to-b from-[#0e0703]/80 to-transparent py-5'
            }`}
        >
            <div className="max-w-6xl mx-auto px-6 flex justify-between items-center h-12">
                <Link
                    href="/"
                    className="flex items-center gap-3 cursor-pointer group"
                >
                    <span className="text-2xl transition-transform group-hover:scale-110">🎣</span>
                    <div className="flex flex-col">
                        <span className="font-pixel text-sm font-bold tracking-tight text-[#f6b535] drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
                            Stardew Fishing AI
                        </span>
                        <span className="text-[10px] text-amber-200/60 font-mono -mt-0.5">
                            video companion
                        </span>
                    </div>
                </Link>

                <div className="hidden md:flex items-center gap-7 text-sm font-medium text-stone-300">
                    <Link href="/play" className="transition-colors hover:text-[#f6b535]">
                        Play Live
                    </Link>
                    <Link href="/evolution" className="transition-colors hover:text-[#f6b535]">
                        AI Evolution
                    </Link>
                    <button 
                        onClick={() => scrollToSection('challenge')} 
                        className="transition-colors hover:text-[#f6b535] cursor-pointer"
                    >
                        $10 Bounty
                    </button>
                    <button 
                        onClick={() => scrollToSection('leaderboard')} 
                        className="transition-colors hover:text-[#f6b535] cursor-pointer"
                    >
                        Leaderboard
                    </button>
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-amber-300/90 transition-colors hover:text-amber-200 font-medium"
                    >
                        Watch Video
                    </a>
                </div>

                <a
                    href={COMPETITION_CONFIG.githubRepo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="stardew-btn-gold px-4 py-2 text-xs font-bold tracking-wide"
                >
                    Starter Repo
                </a>
            </div>
        </motion.nav>
    );
}
