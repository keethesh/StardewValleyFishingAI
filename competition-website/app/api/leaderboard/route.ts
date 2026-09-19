import { getRedis } from '@/lib/redis';
import { NextResponse } from 'next/server';
import { Submission } from '../submissions/route';

export interface LeaderboardEntry {
    rank: number;
    displayName: string;
    score: number;
    catchRate: number;
    modelName: string;
    approach?: string;
    timestamp: number;
}

// GET - Fetch top leaderboard entries
export async function GET() {
    try {
        const redis = await getRedis();

        // Get top 50 from sorted set (highest scores first)
        const topNames = await redis.zRange('leaderboard', 0, 49, { REV: true });

        if (!topNames || topNames.length === 0) {
            return NextResponse.json({
                entries: [
                    { rank: 1, displayName: 'Baseline AI', score: 0.97, catchRate: 0.98, modelName: 'ep3500_dueling_dqn', timestamp: Date.now() },
                    { rank: 2, displayName: 'Your Model Here', score: 0, catchRate: 0, modelName: '???', timestamp: Date.now() },
                ],
                total: 0
            });
        }

        const entries: LeaderboardEntry[] = [];

        for (let i = 0; i < topNames.length; i++) {
            const name = topNames[i] as string;
            const key = `submission:${name.toLowerCase().replace(/\s+/g, '_')}`;
            const data = await redis.get(key);

            if (data) {
                const submission = JSON.parse(data) as Submission;
                entries.push({
                    rank: i + 1,
                    displayName: submission.displayName,
                    score: submission.score,
                    catchRate: submission.catchRate,
                    modelName: submission.modelName,
                    approach: submission.approach,
                    timestamp: submission.timestamp,
                });
            }
        }

        return NextResponse.json({ entries, total: entries.length });

    } catch (error: any) {
        console.error('Leaderboard fetch error:', error);
        return NextResponse.json({
            entries: [
                { rank: 1, displayName: 'Baseline AI', score: 0.97, catchRate: 0.98, modelName: 'ep3500_dueling_dqn', timestamp: Date.now() },
            ],
            total: 0,
            error: 'Redis not configured'
        });
    }
}
