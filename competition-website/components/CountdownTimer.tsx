'use client';

import { useState, useEffect } from 'react';
import { COMPETITION_CONFIG } from '@/lib/competition-config';

interface TimeLeft {
    days: number;
    hours: number;
    minutes: number;
    seconds: number;
}

export default function CountdownTimer() {
    const [timeLeft, setTimeLeft] = useState<TimeLeft | null>(null);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);

        const calculateTimeLeft = (): TimeLeft | null => {
            const endDate = new Date(COMPETITION_CONFIG.endDate);
            const now = new Date();
            const difference = endDate.getTime() - now.getTime();

            if (difference <= 0) {
                return null;
            }

            return {
                days: Math.floor(difference / (1000 * 60 * 60 * 24)),
                hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
                minutes: Math.floor((difference / 1000 / 60) % 60),
                seconds: Math.floor((difference / 1000) % 60),
            };
        };

        setTimeLeft(calculateTimeLeft());

        const timer = setInterval(() => {
            setTimeLeft(calculateTimeLeft());
        }, 1000);

        return () => clearInterval(timer);
    }, []);

    if (!isClient) {
        return <div className="h-20" />; // Placeholder to prevent layout shift
    }

    if (!timeLeft) {
        return (
            <div className="bg-red-500/20 border border-red-500/50 rounded-xl px-6 py-4 text-red-400 font-bold">
                Competition Ended!
            </div>
        );
    }

    return (
        <div className="flex justify-center gap-4">
            {[
                { value: timeLeft.days, label: 'Days' },
                { value: timeLeft.hours, label: 'Hours' },
                { value: timeLeft.minutes, label: 'Min' },
                { value: timeLeft.seconds, label: 'Sec' },
            ].map(({ value, label }) => (
                <div key={label} className="bg-slate-800/50 border border-white/10 rounded-lg px-4 py-3 min-w-[70px]">
                    <div className="text-2xl md:text-3xl font-bold text-white font-mono">
                        {String(value).padStart(2, '0')}
                    </div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider">{label}</div>
                </div>
            ))}
        </div>
    );
}
