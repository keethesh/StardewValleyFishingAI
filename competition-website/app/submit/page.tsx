'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Upload, AlertCircle, CheckCircle } from 'lucide-react';
import { InferenceSession } from 'onnxruntime-web';

function getErrorMessage(error: unknown, fallback: string) {
    return error instanceof Error ? error.message : fallback;
}

type OfficialResult = {
    score: number;
    catchRate: number;
    scoreHard: number;
    episodes: number;
    nFish: number;
    seedsPerFish: number;
    runSeed: number;
};

export default function SubmitPage() {
    const [file, setFile] = useState<File | null>(null);
    const [modelName, setModelName] = useState('');
    const [author, setAuthor] = useState('');

    // Submission State
    const [status, setStatus] = useState<'idle' | 'uploading' | 'evaluating' | 'success' | 'error'>('idle');
    const [errorMsg, setErrorMsg] = useState('');
    const [evalResult, setEvalResult] = useState<OfficialResult | null>(null);

    // Evaluation progress
    const [evalProgress, setEvalProgress] = useState({ current: 0, total: 0, fishName: '' });

    // State for verification
    const [verificationStatus, setVerificationStatus] = useState<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
    const [verificationMsg, setVerificationMsg] = useState('');
    const [verifiedSession, setVerifiedSession] = useState<InferenceSession | null>(null);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const f = e.target.files[0];
            if (!f.name.endsWith('.onnx')) {
                alert('Please select a .onnx file');
                return;
            }
            setFile(f);
            await verifyModel(f);
        }
    };

    const verifyModel = async (file: File) => {
        setVerificationStatus('checking');
        setVerificationMsg('Verifying architecture compatibility...');

        try {
            // dynamic import to avoid SSR issues if any
            const ort = await import('onnxruntime-web');

            // Suppress benign warnings (e.g., CPU vendor check)
            ort.env.logLevel = 'error';

            const url = URL.createObjectURL(file);
            const session = await ort.InferenceSession.create(url, {
                executionProviders: ['wasm'], // Use WASM only for stable pre-flight check
                graphOptimizationLevel: 'basic'
            });

            // check inputs
            if (!session.inputNames.includes('state')) {
                throw new Error(`Invalid Input Name. Expected 'state', found: ${session.inputNames.join(', ')}`);
            }

            // run dummy inference — competition contract is 8-D
            const dummyInput = new Float32Array(8).fill(0.5);
            const tensor = new ort.Tensor('float32', dummyInput, [1, 8]);

            const results = await session.run({ state: tensor });

            // check outputs
            if (!results.q_values) {
                const outputNames = Object.keys(results);
                throw new Error(`Invalid Output Name. Expected 'q_values', found: ${outputNames.join(', ')}`);
            }

            // check output shape (approx)
            const output = results.q_values;
            if (output.dims[1] !== 2) {
                throw new Error(`Invalid Output Shape. Expected [1, 2], found ${output.dims.join('x')}`);
            }

            setVerificationStatus('valid');
            setVerificationMsg('✅ Model architecture is valid!');
            setVerifiedSession(session); // Save session for evaluation

            // Don't revoke URL yet - we need the session for evaluation

        } catch (error: unknown) {
            console.error(error);
            setVerificationStatus('invalid');
            setVerificationMsg(`❌ Verification Failed: ${getErrorMessage(error, 'Unknown verification error')}`);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file || !modelName || !author || verificationStatus !== 'valid') return;

        setStatus('evaluating');
        setErrorMsg('');
        setEvalResult(null);
        setEvalProgress({
            current: 0,
            total: 64 * 3,
            fishName: 'Full catalog × 3 seeds…',
        });

        try {
            // Official: every fish × 3 seeds, fresh runSeed (not training eval seeds)
            const form = new FormData();
            form.append('model', file);

            const evalRes = await fetch('/api/evaluate', {
                method: 'POST',
                body: form,
            });
            const evalJson = await evalRes.json();
            if (!evalRes.ok || !evalJson.success) {
                throw new Error(evalJson.error || 'Official evaluation failed');
            }

            const r = evalJson.results as OfficialResult;
            setEvalResult(r);
            setEvalProgress({
                current: r.episodes,
                total: r.episodes,
                fishName: `runSeed ${r.runSeed}`,
            });

            try {
                await fetch('/api/submissions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        displayName: author,
                        modelName: modelName,
                        score: r.score,
                        catchRate: r.catchRate,
                    }),
                });
            } catch {
                console.log('Could not save to leaderboard (KV may not be configured)');
            }

            setStatus('success');
        } catch (error: unknown) {
            setStatus('error');
            setErrorMsg(getErrorMessage(error, 'Evaluation failed'));
        }
    };

    return (
        <main className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-950 via-slate-950 to-black text-slate-200 font-sans p-8 selection:bg-cyan-500 selection:text-white">

            {/* Background Decor */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-[-20%] left-[20%] w-[50%] h-[50%] bg-blue-600/10 rounded-full blur-[150px]" />
            </div>

            <div className="max-w-3xl mx-auto space-y-8 relative z-10">
                <Link href="/" className="inline-flex items-center gap-2 text-slate-400 transition-colors hover:text-white">
                    ← Back to Companion Site
                </Link>

                <div className="text-center space-y-2">
                    <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
                        Submit a Model
                    </h1>
                    <p className="text-slate-400 text-lg">
                        Upload your <code className="bg-slate-800 px-2 py-0.5 rounded text-cyan-400">.onnx</code> file if you want to take a shot at the challenge.
                    </p>
                </div>

                <div className="bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl ring-1 ring-white/5">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid md:grid-cols-2 gap-6">
                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Model Name</label>
                                <input
                                    type="text"
                                    required
                                    value={modelName}
                                    onChange={e => setModelName(e.target.value)}
                                    className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all placeholder:text-slate-600"
                                    placeholder="e.g. DeepFisher v1"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Display Name</label>
                                <input
                                    type="text"
                                    required
                                    value={author}
                                    onChange={e => setAuthor(e.target.value)}
                                    className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all placeholder:text-slate-600"
                                    placeholder="Leaderboard Name"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Contact (Private)</label>
                                <input
                                    type="text"
                                    required
                                    className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all placeholder:text-slate-600"
                                    placeholder="Discord ID or Email"
                                />
                            </div>

                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                                    GitHub URL <span className="text-slate-600 lowercase font-normal">(optional)</span>
                                </label>
                                <input
                                    type="url"
                                    className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all placeholder:text-slate-600"
                                    placeholder="https://github.com/..."
                                />
                            </div>

                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                                    Brief Approach <span className="text-slate-600 lowercase font-normal">(optional)</span>
                                </label>
                                <textarea
                                    rows={2}
                                    className="w-full bg-black/20 border border-white/10 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all placeholder:text-slate-600 resize-none"
                                    placeholder="e.g. PPO with reward shaping, 2M timesteps..."
                                />
                            </div>
                        </div>

                        <div className="relative group">
                            <div className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer relative overflow-hidden
                        ${file ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-white/10 hover:border-cyan-500/50 hover:bg-white/5'}`}>

                                <input
                                    type="file"
                                    required
                                    accept=".onnx"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                                />

                                <div className="pointer-events-none relative z-10">
                                    {file ? (
                                        <div className="space-y-2 animate-in fade-in zoom-in duration-300">
                                            <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                                                <CheckCircle className="w-8 h-8 text-emerald-400" />
                                            </div>
                                            <p className="text-emerald-300 font-medium text-lg">{file.name}</p>
                                            <p className="text-slate-500 text-sm">{(file.size / 1024).toFixed(0)} KB • Ready to upload</p>
                                        </div>
                                    ) : (
                                        <div className="space-y-2">
                                            <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                                                <Upload className="w-8 h-8 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                                            </div>
                                            <p className="text-slate-300 font-medium text-lg">Drag & drop your model here</p>
                                            <p className="text-slate-500 text-sm">or click to browse files</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Verification Status Message */}
                        {verificationStatus !== 'idle' && (
                            <div className={`p-4 rounded-lg text-sm font-mono border ${verificationStatus === 'valid' ? 'bg-emerald-900/30 border-emerald-500/30 text-emerald-300' : verificationStatus === 'invalid' ? 'bg-red-900/30 border-red-500/30 text-red-300' : 'bg-slate-800 border-slate-700 text-slate-300'}`}>
                                {verificationStatus === 'checking' && <span className="animate-pulse">⏳ </span>}
                                {verificationMsg}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={status === 'uploading' || status === 'evaluating' || verificationStatus !== 'valid'}
                            className={`w-full py-4 rounded-xl font-bold text-lg transition-all shadow-lg relative overflow-hidden group
                        ${(status === 'idle' && verificationStatus === 'valid') || status === 'error' ? 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:scale-[1.02] text-white' : 'bg-slate-800 text-slate-500 cursor-not-allowed opacity-50'}`}
                        >
                            <span className="relative z-10 flex items-center justify-center gap-2">
                                {status === 'idle' && '🚀 Launch Evaluation'}
                                {status === 'uploading' && 'Uploading...'}
                                {status === 'evaluating' && 'Running Simulation...'}
                                {status === 'success' && 'Done!'}
                                {status === 'error' && 'Retry Submission'}
                            </span>
                            {status === 'idle' && verificationStatus === 'valid' && <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />}
                        </button>
                    </form>

                    {/* Progress during evaluation */}
                    {status === 'evaluating' && (
                        <div className="mt-8 bg-slate-800/50 border border-slate-700 p-6 rounded-xl">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-slate-300 font-medium">Evaluating: {evalProgress.fishName}</span>
                                <span className="text-cyan-400 font-mono">{evalProgress.current}/{evalProgress.total}</span>
                            </div>
                            <div className="w-full bg-slate-700 rounded-full h-3">
                                <div
                                    className="bg-gradient-to-r from-cyan-500 to-blue-500 h-3 rounded-full transition-all duration-300"
                                    style={{ width: `${evalProgress.total > 0 ? (evalProgress.current / evalProgress.total) * 100 : 0}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {status === 'success' && evalResult && (
                        <div className="mt-8 bg-emerald-900/30 border border-emerald-500/30 p-6 rounded-xl animate-in slide-in-from-bottom-4">
                            <div className="flex items-center gap-4 mb-4">
                                <div className="p-3 bg-emerald-500/20 rounded-full">
                                    <CheckCircle className="text-emerald-400 w-8 h-8" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-emerald-400 text-lg">Official evaluation complete</h4>
                                    <span className="text-teal-300 text-xs font-mono">
                                        All {evalResult.nFish} fish × {evalResult.seedsPerFish}{' '}
                                        seeds · seed {evalResult.runSeed}
                                    </span>
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-4 text-center mt-4">
                                <div className="bg-black/20 p-4 rounded-lg">
                                    <div className="text-3xl font-bold text-white">{(evalResult.score * 100).toFixed(1)}</div>
                                    <div className="text-xs text-slate-400 uppercase tracking-wider">Score %</div>
                                </div>
                                <div className="bg-black/20 p-4 rounded-lg">
                                    <div className="text-3xl font-bold text-cyan-400">{(evalResult.catchRate * 100).toFixed(0)}%</div>
                                    <div className="text-xs text-slate-400 uppercase tracking-wider">Catch Rate</div>
                                </div>
                                <div className="bg-black/20 p-4 rounded-lg">
                                    <div className="text-3xl font-bold text-purple-400">{(evalResult.scoreHard * 100).toFixed(0)}%</div>
                                    <div className="text-xs text-slate-400 uppercase tracking-wider">Hard Score</div>
                                </div>
                            </div>

                            <p className="text-xs text-slate-500 mt-4 text-center italic">
                                Leaderboard score is difficulty-weighted catch rate on every fish
                                (3 fresh seeds each).
                            </p>
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="mt-8 bg-red-900/30 border border-red-500/30 p-6 rounded-xl flex items-center gap-4 animate-in slide-in-from-bottom-4">
                            <div className="p-3 bg-red-500/20 rounded-full">
                                <AlertCircle className="text-red-400 w-8 h-8" />
                            </div>
                            <div>
                                <h4 className="font-bold text-red-400 text-lg">Submission Failed</h4>
                                <p className="text-red-200/80">{errorMsg}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}

