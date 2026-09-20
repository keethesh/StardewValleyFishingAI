'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { COMPETITION_CONFIG } from '@/lib/competition-config';

const NAV_LINKS = [
    { href: '/play', label: 'Play' },
    { href: '/evolution', label: 'Evolution' },
    { href: '/submit', label: 'Compete' },
    { href: '/#leaderboard', label: 'Leaderboard' },
] as const;

function navLinkClass(active: boolean) {
    return `whitespace-nowrap transition-colors hover:text-[#f6b535] ${
        active ? 'text-[#f6b535]' : 'text-stone-300'
    }`;
}

export default function Navbar() {
    const pathname = usePathname();
    const [scrolled, setScrolled] = useState(false);
    const [hash, setHash] = useState('');
    const videoUrl = `https://www.youtube.com/watch?v=${COMPETITION_CONFIG.youtubeVideoId}`;

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        handleScroll();
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    useEffect(() => {
        const updateHash = () => setHash(window.location.hash);
        updateHash();
        window.addEventListener('hashchange', updateHash);
        return () => window.removeEventListener('hashchange', updateHash);
    }, [pathname]);

    const isActive = (href: string) => {
        if (href === '/#leaderboard') {
            return pathname === '/' && hash === '#leaderboard';
        }
        if (href === '/submit') {
            return pathname === '/submit' || pathname === '/rules';
        }
        return pathname === href;
    };

    return (
        <motion.nav
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.4 }}
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-200 ${
                scrolled
                    ? 'border-b-2 border-[#5c3014] bg-[#161c32]/95 py-3 shadow-[0_6px_24px_rgba(0,0,0,0.7),inset_0_-1px_0_rgba(247,191,71,0.25)] backdrop-blur-md'
                    : 'bg-gradient-to-b from-[#111526]/85 via-[#111526]/40 to-transparent py-5'
            }`}
        >
            <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-2 px-3 sm:px-6">
                <Link href="/" className="group flex shrink-0 cursor-pointer items-center gap-2 sm:gap-3">
                    <span className="text-xl transition-transform group-hover:scale-110 sm:text-2xl">🎣</span>
                    <div className="hidden flex-col sm:flex">
                        <span className="font-pixel text-sm font-bold tracking-tight text-[#f6b535] drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
                            Stardew Fishing AI
                        </span>
                        <span className="-mt-0.5 font-mono text-[10px] text-amber-200/60">video companion</span>
                    </div>
                </Link>

                <div className="flex min-w-0 flex-1 items-center justify-center gap-2 overflow-x-auto px-1 text-[10px] font-medium sm:gap-4 sm:text-xs md:gap-6 md:text-sm">
                    {NAV_LINKS.map(({ href, label }) => (
                        <Link key={href} href={href} className={navLinkClass(isActive(href))}>
                            {label}
                        </Link>
                    ))}
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hidden whitespace-nowrap text-amber-300/90 transition-colors hover:text-amber-200 md:inline"
                    >
                        Watch Video
                    </a>
                </div>

                <a
                    href={COMPETITION_CONFIG.githubRepo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="stardew-btn-gold hidden shrink-0 px-3 py-2 text-[10px] font-bold tracking-wide sm:inline-flex sm:px-4 sm:text-xs"
                >
                    Starter Repo
                </a>
            </div>
        </motion.nav>
    );
}
