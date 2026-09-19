'use client';

import { useState, useEffect } from 'react';
import { Trophy } from 'lucide-react';

interface LeaderboardEntry {
  rank: number;
  displayName: string;
  modelName: string;
  score: number;
  catchRate: number;
  timestamp: number;
}

const FALLBACK_DATA: LeaderboardEntry[] = [
  { rank: 1, displayName: "Baseline AI", modelName: "ep3500_dueling_dqn", score: 0.97, catchRate: 0.98, timestamp: Date.now() },
  { rank: 2, displayName: "Your Name Here", modelName: "train_something_better", score: 0, catchRate: 0, timestamp: Date.now() },
];

export default function Leaderboard() {
  const [entries, setEntries] = useState<LeaderboardEntry[]>(FALLBACK_DATA);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        const res = await fetch('/api/leaderboard');
        const data = await res.json();
        if (data.entries && data.entries.length > 0) {
          setEntries(data.entries);
        }
      } catch {
        console.log('Using fallback leaderboard data');
      } finally {
        setLoading(false);
      }
    }
    fetchLeaderboard();
  }, []);

  return (
    <div className="mx-auto w-full max-w-5xl overflow-hidden rounded-[28px] border border-white/10 bg-stone-950/55 shadow-2xl ring-1 ring-white/5">
      <div className="flex items-center justify-between border-b border-white/5 bg-white/5 p-6">
        <div className="space-y-1">
          <h2 className="flex items-center gap-3 text-xl font-bold text-stone-100">
            <Trophy className="h-6 w-6 text-amber-300" />
            Challenge Leaderboard
          </h2>
          <p className="text-sm text-stone-400">
            The first benchmark is the model from the video. Everything above that is community work.
          </p>
        </div>
        <span className="rounded-full border border-white/10 px-3 py-1 text-[11px] font-mono uppercase tracking-[0.24em] text-stone-500">
          {loading ? 'Loading' : 'Live'}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead className="bg-black/20 text-xs font-bold uppercase tracking-wider text-stone-400">
            <tr>
              <th className="px-6 py-4">Rank</th>
              <th className="px-6 py-4">Model</th>
              <th className="px-6 py-4">Builder</th>
              <th className="px-6 py-4">Catch Rate</th>
              <th className="px-6 py-4 text-right">Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {entries.map((entry) => {
              let rankDisplay: React.ReactNode = <span className="font-mono text-stone-500">#{entry.rank}</span>;
              if (entry.rank === 1) rankDisplay = <span className="text-2xl">🥇</span>;
              if (entry.rank === 2) rankDisplay = <span className="text-2xl">🥈</span>;
              if (entry.rank === 3) rankDisplay = <span className="text-2xl">🥉</span>;

              const rowGlow = entry.rank === 1 ? "bg-amber-300/5 hover:bg-amber-300/10" : "hover:bg-white/5";
              const scoreColor = entry.rank === 1 ? "text-amber-300" : "text-teal-200";

              return (
                <tr key={entry.rank} className={`transition-colors duration-200 ${rowGlow}`}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex h-8 w-8 items-center justify-center">
                      {rankDisplay}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="block font-medium text-white">{entry.modelName}</span>
                    <span className="text-xs text-stone-500">
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-stone-300">
                    <div className="flex items-center gap-2">
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-stone-800 text-xs uppercase text-white">
                        {entry.displayName[0]}
                      </div>
                      {entry.displayName}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-stone-800">
                        <div
                          className="h-full rounded-full bg-emerald-400"
                          style={{ width: `${entry.catchRate * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono text-emerald-300">
                        {(entry.catchRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className={`px-6 py-4 text-right font-mono text-lg font-bold ${scoreColor}`}>
                    {(entry.score <= 1 ? entry.score * 100 : entry.score).toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
