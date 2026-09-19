import Link from 'next/link';
import { COMPETITION_CONFIG } from '@/lib/competition-config';

export default function Footer() {
    const videoUrl = `https://www.youtube.com/watch?v=${COMPETITION_CONFIG.youtubeVideoId}`;

    return (
        <footer className="border-t-2 border-[#5c3214] bg-[#0d0704] py-12 text-stone-300">
            <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 md:flex-row md:items-end md:justify-between">
                <div className="text-center md:text-left">
                    <div className="mb-2 flex items-center justify-center gap-2 md:justify-start">
                        <span className="text-2xl">🎣</span>
                        <span className="font-pixel text-sm font-bold text-[#f6b535]">Stardew Fishing AI</span>
                    </div>
                    <p className="text-xs leading-6 text-amber-200/50 font-mono">
                        Companion site for the Stardew Valley fishing AI video breakdown.
                        <br />
                        Trained with Dueling Double DQN on Newtonian 1-D physics.
                        <br />
                        Not affiliated with ConcernedApe or Stardew Valley.
                    </p>
                </div>

                <div className="flex flex-wrap justify-center gap-6 text-xs font-pixel text-amber-200/70 md:justify-end">
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors hover:text-[#f6b535]"
                    >
                        Video
                    </a>
                    <a
                        href={COMPETITION_CONFIG.githubRepo}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors hover:text-[#f6b535]"
                    >
                        Starter Repo
                    </a>
                    <Link href="/play" className="transition-colors hover:text-[#f6b535]">Play Live</Link>
                    <Link href="/evolution" className="transition-colors hover:text-[#f6b535]">Evolution</Link>
                    <Link href="/submit" className="transition-colors hover:text-[#f6b535]">Submit Model</Link>
                    <Link href="/rules" className="transition-colors hover:text-[#f6b535]">Regulations</Link>
                </div>
            </div>
        </footer>
    );
}
