import { NextResponse } from 'next/server';
import * as ort from 'onnxruntime-node';
import { runOfficialEvaluation, OFFICIAL_SEEDS_PER_FISH } from '@/lib/simulation';

export const maxDuration = 60;
export const dynamic = 'force-dynamic';

const MAX_MODEL_BYTES = 5 * 1024 * 1024; // competition contract

async function loadModelBuffer(req: Request): Promise<ArrayBuffer> {
  const contentType = req.headers.get('content-type') || '';

  if (contentType.includes('multipart/form-data')) {
    const form = await req.formData();
    const file = form.get('model');
    if (!(file instanceof File)) {
      throw new Error('Missing model file field "model"');
    }
    if (file.size > MAX_MODEL_BYTES) {
      throw new Error(`Model exceeds ${MAX_MODEL_BYTES} byte limit`);
    }
    return file.arrayBuffer();
  }

  const body = await req.json();
  if (body.modelBase64 && typeof body.modelBase64 === 'string') {
    const buf = Buffer.from(body.modelBase64, 'base64');
    if (buf.byteLength > MAX_MODEL_BYTES) {
      throw new Error(`Model exceeds ${MAX_MODEL_BYTES} byte limit`);
    }
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  if (body.modelUrl && typeof body.modelUrl === 'string') {
    let url = body.modelUrl as string;
    if (!url.startsWith('http')) {
      const host = req.headers.get('host') || 'localhost:3000';
      const protocol = host.includes('localhost') ? 'http' : 'https';
      url = `${protocol}://${host}${url}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch model: ${res.statusText}`);
    const buf = await res.arrayBuffer();
    if (buf.byteLength > MAX_MODEL_BYTES) {
      throw new Error(`Model exceeds ${MAX_MODEL_BYTES} byte limit`);
    }
    return buf;
  }

  throw new Error('Provide model file, modelBase64, or modelUrl');
}

/**
 * Official leaderboard evaluation.
 * Full catalog × N seeds, fresh runSeed — not the public training eval seed schedule.
 */
export async function POST(req: Request) {
  try {
    const modelBuffer = await loadModelBuffer(req);
    const session = await ort.InferenceSession.create(modelBuffer);

    const dummy = new ort.Tensor('float32', new Float32Array(8), [1, 8]);
    const smoke = await session.run({ state: dummy });
    if (!smoke.q_values || (smoke.q_values.dims[1] ?? 0) !== 2) {
      return NextResponse.json(
        { error: 'Invalid model contract: expected state[1,8] → q_values[1,2]' },
        { status: 400 }
      );
    }

    const results = await runOfficialEvaluation(session, {
      seedsPerFish: OFFICIAL_SEEDS_PER_FISH,
    });

    return NextResponse.json({
      success: true,
      mode: 'official',
      results: {
        score: results.score,
        catchRate: results.catchRate,
        scoreHard: results.scoreHard,
        nFish: results.nFish,
        seedsPerFish: results.seedsPerFish,
        episodes: results.episodes,
        avgSteps: results.avgSteps,
        runSeed: results.runSeed,
        byBehaviour: results.byBehaviour,
      },
    });
  } catch (error: unknown) {
    console.error('Official evaluation failed:', error);
    const message = error instanceof Error ? error.message : 'Evaluation failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
