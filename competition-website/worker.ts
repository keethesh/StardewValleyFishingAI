/// <reference types="@cloudflare/workers-types" />

interface Env {
  ASSETS: Fetcher;
  LEADERBOARD_KV?: KVNamespace;
}

interface Submission {
  displayName: string;
  modelName: string;
  score: number;
  catchRate: number;
  submittedAt: string;
}

const BASELINE_SUBMISSION: Submission = {
  displayName: 'Baseline AI (Episode 3500)',
  modelName: 'baseline.onnx',
  score: 0.97,
  catchRate: 1.0,
  submittedAt: '2026-06-02T08:00:00Z',
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // API: Leaderboard
    if (url.pathname === '/api/leaderboard' && request.method === 'GET') {
      try {
        let submissions: Submission[] = [];
        if (env.LEADERBOARD_KV) {
          const raw = await env.LEADERBOARD_KV.get('submissions', { type: 'json' });
          if (Array.isArray(raw)) {
            submissions = raw as Submission[];
          }
        }
        if (submissions.length === 0) {
          submissions = [BASELINE_SUBMISSION];
        } else if (!submissions.some((s) => s.displayName.includes('Baseline AI'))) {
          submissions.push(BASELINE_SUBMISSION);
        }

        // Sort by score descending
        submissions.sort((a, b) => b.score - a.score);

        const leaderboard = submissions.map((sub, idx) => ({
          rank: idx + 1,
          ...sub,
        }));

        return new Response(JSON.stringify({ success: true, leaderboard }), {
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'public, s-maxage=10, stale-while-revalidate=30',
          },
        });
      } catch (err: unknown) {
        console.error('Leaderboard fetch error:', err);
        return new Response(JSON.stringify({ error: 'Failed to fetch leaderboard' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    // API: Submissions
    if (url.pathname === '/api/submissions' && request.method === 'POST') {
      try {
        const body = (await request.json()) as Partial<Submission>;
        if (!body.displayName || typeof body.score !== 'number') {
          return new Response(JSON.stringify({ error: 'Invalid submission data' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' },
          });
        }

        const newSubmission: Submission = {
          displayName: String(body.displayName).slice(0, 32),
          modelName: String(body.modelName || 'MyModel').slice(0, 32),
          score: Number(body.score),
          catchRate: Number(body.catchRate || 0),
          submittedAt: new Date().toISOString(),
        };

        if (env.LEADERBOARD_KV) {
          const raw = await env.LEADERBOARD_KV.get('submissions', { type: 'json' });
          const list: Submission[] = Array.isArray(raw) ? (raw as Submission[]) : [];
          list.push(newSubmission);
          // Keep top 100
          list.sort((a, b) => b.score - a.score);
          await env.LEADERBOARD_KV.put('submissions', JSON.stringify(list.slice(0, 100)));
        }

        return new Response(JSON.stringify({ success: true, submission: newSubmission }), {
          status: 201,
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (err: unknown) {
        console.error('Submission save error:', err);
        return new Response(JSON.stringify({ error: 'Failed to save submission' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    // Static Assets
    const response = await env.ASSETS.fetch(request);

    // Attach Cross-Origin-Opener-Policy & Cross-Origin-Embedder-Policy headers for WASM
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
