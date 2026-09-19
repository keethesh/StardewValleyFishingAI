import Link from 'next/link';
import { COMPETITION_CONFIG } from '@/lib/competition-config';

export default function Footer() {
    const videoUrl = `https://www.youtube.com/watch?v=${COMPETITION_CONFIG.youtubeVideoId}`;

    return (
        <footer className="border-t border-white/10 bg-slate-950/50 py-12">
            <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 md:flex-row md:items-end md:justify-between">
                <div className="text-center md:text-left">
                    <div className="mb-2 flex items-center justify-center gap-2 md:justify-start">
                        <span className="text-2xl">🎣</span>
                        <span className="text-lg font-semibold text-stone-100">Stardew Fishing AI</span>
                    </div>
                    <p className="text-sm leading-6 text-stone-500">
                        Companion site for the Stardew fishing AI video.
                        <br />
                        Play the demo, poke at the benchmark, then train your own if you want.
                        <br />
                        Not affiliated with ConcernedApe or Stardew Valley.
                    </p>
                </div>

                <div className="flex flex-wrap justify-center gap-6 text-sm font-medium text-stone-400 md:justify-end">
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors hover:text-white"
                    >
                        Watch Video
                    </a>
                    <a
                        href={COMPETITION_CONFIG.githubRepo}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors hover:text-white"
                    >
                        Starter Repo
                    </a>
                    <Link href="/submit" className="transition-colors hover:text-white">Submit Model</Link>
                    <Link href="/rules" className="transition-colors hover:text-white">Challenge Guide</Link>
                </div>
            </div>
        </footer>
    );
}
