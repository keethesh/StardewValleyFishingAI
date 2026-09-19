import { getRedis } from '@/lib/redis';
import { NextRequest, NextResponse } from 'next/server';

export interface Submission {
    displayName: string;
    modelName: string;
    score: number;
    catchRate: number;
    approach?: string;
    githubUrl?: string;
    timestamp: number;
}

// POST - Save a new submission
export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        const { displayName, modelName, score, catchRate, approach, githubUrl } = body;

        if (!displayName || !modelName || typeof score !== 'number') {
            return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
        }

        const submission: Submission = {
            displayName,
            modelName,
            score,
            catchRate: catchRate || 0,
            approach: approach || '',
            githubUrl: githubUrl || '',
            timestamp: Date.now(),
        };

        const redis = await getRedis();

        // Use displayName as key (allows updates)
        const key = `submission:${displayName.toLowerCase().replace(/\s+/g, '_')}`;

        // Save full submission data
        await redis.set(key, JSON.stringify(submission));

        // Update sorted leaderboard (for ranking)
        await redis.zAdd('leaderboard', { score, value: displayName });

        return NextResponse.json({ success: true, submission });

    } catch (error: any) {
        console.error('Submission error:', error);
        return NextResponse.json({ error: error.message || 'Failed to save submission' }, { status: 500 });
    }
}
