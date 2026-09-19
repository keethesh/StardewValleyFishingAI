'use client';

import { useState, useEffect } from 'react';

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
        if (res.ok) {
          const data = await res.json();
          if (data.entries && data.entries.length > 0) {
            setEntries(data.entries);
          }
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
    <div className="mx-auto w-full max-w-5xl stardew-box overflow-hidden">
      <div className="flex items-center justify-between border-b-2 border-[#5c3214] bg-[#1a0e07] p-5">
        <div className="space-y-1">
          <p className="font-pixel text-xs text-[#f6b535] uppercase tracking-wider">
            Town Leaderboard
          </p>
          <p className="text-xs text-amber-200/60 font-mono">
            Evaluated on the hidden 25-fish test suite
          </p>
        </div>
        <span className="font-pixel text-[10px] text-[#f6b535] bg-[#331a0b] px-3 py-1.5 border border-[#6b3813] rounded">
          {loading ? 'Checking...' : 'Live Records'}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead className="bg-[#120904] text-[11px] font-pixel uppercase tracking-wider text-amber-200/70 border-b border-[#3d200d]">
            <tr>
              <th className="px-6 py-3.5">Rank</th>
              <th className="px-6 py-3.5">Model</th>
              <th className="px-6 py-3.5">Angler</th>
              <th className="px-6 py-3.5">Catch Rate</th>
              <th className="px-6 py-3.5 text-right">Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#3d200d]/60 bg-[#170d07]">
            {entries.map((entry) => {
              let rankDisplay: React.ReactNode = <span className="font-pixel text-xs text-stone-400">#{entry.rank}</span>;
              if (entry.rank === 1) rankDisplay = <span className="text-xl">🥇</span>;
              if (entry.rank === 2) rankDisplay = <span className="text-xl">🥈</span>;
              if (entry.rank === 3) rankDisplay = <span className="text-xl">🥉</span>;

              const rowGlow = entry.rank === 1 ? "bg-[#29170a] hover:bg-[#341d0c]" : "hover:bg-[#201107]";
              const scoreColor = entry.rank === 1 ? "text-[#f6b535]" : "text-amber-100";

              return (
                <tr key={entry.rank} className={`transition-colors duration-150 ${rowGlow}`}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex h-7 w-7 items-center justify-center">
                      {rankDisplay}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="block font-medium text-stone-100">{entry.modelName}</span>
                    <span className="text-xs text-amber-200/40 font-mono">
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-stone-300">
                    <div className="flex items-center gap-2">
                      <div className="flex h-6 w-6 items-center justify-center rounded bg-[#42220d] border border-[#6b3813] text-xs font-bold uppercase text-[#f6b535]">
                        {entry.displayName[0]}
                      </div>
                      <span className="text-sm font-medium">{entry.displayName}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="h-2 w-20 overflow-hidden rounded bg-[#0a0502] border border-[#3d200d]">
                        <div
                          className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400"
                          style={{ width: `${entry.catchRate * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono font-bold text-emerald-300">
                        {(entry.catchRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className={`px-6 py-4 text-right font-mono text-base font-bold ${scoreColor}`}>
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
