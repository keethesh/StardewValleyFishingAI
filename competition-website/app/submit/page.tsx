'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Upload, AlertCircle, CheckCircle } from 'lucide-react';
import { InferenceSession } from 'onnxruntime-web';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { runOfficialEvaluation } from '@/lib/simulation';

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
        if (!file || !modelName || !author || verificationStatus !== 'valid' || !verifiedSession) return;

        setStatus('evaluating');
        setErrorMsg('');
        setEvalResult(null);
        setEvalProgress({
            current: 0,
            total: 25 * 3,
            fishName: 'Starting official evaluation…',
        });

        try {
            // Run official evaluation client-side with onnxruntime-web
            const r = await runOfficialEvaluation(verifiedSession, {
                onProgress: (info) => {
                    setEvalProgress(info);
                },
            });
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
        <main className="min-h-screen text-[#f7f2ea]">
            <Navbar />
            <div className="max-w-3xl mx-auto space-y-8 relative z-10 px-6 pb-24 pt-28 md:pt-32">
                <Link href="/" className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#f7bf47] hover:text-[#ffdb80] transition-colors">
                    ← Back to Companion Demo
                </Link>

                <div className="text-center space-y-3">
                    <p className="text-xs font-bold uppercase tracking-[0.32em] text-[#f7bf47]">
                        Community Challenge
                    </p>
                    <h1 className="font-[family-name:var(--font-pixel)] text-3xl sm:text-4xl text-[#fff7e6] drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
                        Submit a Model
                    </h1>
                    <p className="text-[#d8cbba] text-sm sm:text-base max-w-xl mx-auto leading-relaxed">
                        Drop your trained <code className="bg-[#180e07] border border-[#522a0e] px-2 py-0.5 rounded text-[#f7bf47] font-mono text-xs">.onnx</code> file below to run the official evaluation benchmark.
                    </p>
                    <p className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-[#c4b5a3]">
                        <Link href="/rules" className="font-bold uppercase tracking-wider text-[#f7bf47] hover:text-[#ffdb80] transition-colors">
                            Challenge rules
                        </Link>
                        <Link href="/#leaderboard" className="font-bold uppercase tracking-wider text-[#f7bf47] hover:text-[#ffdb80] transition-colors">
                            View leaderboard
                        </Link>
                    </p>
                </div>

                <div className="stardew-box p-6 sm:p-8 shadow-2xl">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid md:grid-cols-2 gap-6">
                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">Model Name</label>
                                <input
                                    type="text"
                                    required
                                    value={modelName}
                                    onChange={e => setModelName(e.target.value)}
                                    className="w-full bg-[#180e07] border-2 border-[#6b3813] rounded p-3 text-[#fff7e6] focus:border-[#f7bf47] outline-none transition-all placeholder:text-[#6a5441] text-sm"
                                    placeholder="e.g. DeepFisher v1"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">Display Name</label>
                                <input
                                    type="text"
                                    required
                                    value={author}
                                    onChange={e => setAuthor(e.target.value)}
                                    className="w-full bg-[#180e07] border-2 border-[#6b3813] rounded p-3 text-[#fff7e6] focus:border-[#f7bf47] outline-none transition-all placeholder:text-[#6a5441] text-sm"
                                    placeholder="Leaderboard Handle"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">Contact (Private)</label>
                                <input
                                    type="text"
                                    required
                                    className="w-full bg-[#180e07] border-2 border-[#6b3813] rounded p-3 text-[#fff7e6] focus:border-[#f7bf47] outline-none transition-all placeholder:text-[#6a5441] text-sm"
                                    placeholder="Discord ID or Email (for bounty payout)"
                                />
                            </div>

                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">
                                    GitHub / Colab URL <span className="text-[#a89682] lowercase font-normal">(optional)</span>
                                </label>
                                <input
                                    type="url"
                                    className="w-full bg-[#180e07] border-2 border-[#6b3813] rounded p-3 text-[#fff7e6] focus:border-[#f7bf47] outline-none transition-all placeholder:text-[#6a5441] text-sm"
                                    placeholder="https://github.com/..."
                                />
                            </div>

                            <div className="md:col-span-2">
                                <label className="block text-xs font-bold uppercase tracking-wider text-[#e6b978] mb-2 font-mono">
                                    Training Strategy <span className="text-[#a89682] lowercase font-normal">(optional)</span>
                                </label>
                                <textarea
                                    rows={2}
                                    className="w-full bg-[#180e07] border-2 border-[#6b3813] rounded p-3 text-[#fff7e6] focus:border-[#f7bf47] outline-none transition-all placeholder:text-[#6a5441] resize-none text-sm"
                                    placeholder="e.g. Dueling DQN, 5000 episodes, tuned reward shaping..."
                                />
                            </div>
                        </div>
                        <div className="relative group">
                            <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer relative overflow-hidden
                        ${file ? 'border-[#a3e635] bg-[#a3e635]/10' : 'border-[#b86e33] bg-[#180e07] hover:border-[#f7bf47]'}`}>

                                <input
                                    type="file"
                                    required
                                    accept=".onnx"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                                />

                                <div className="pointer-events-none relative z-10">
                                    {file ? (
                                        <div className="space-y-2">
                                            <div className="w-12 h-12 bg-[#a3e635]/20 rounded-full flex items-center justify-center mx-auto mb-2">
                                                <CheckCircle className="w-6 h-6 text-[#a3e635]" />
                                            </div>
                                            <p className="text-[#a3e635] font-bold text-base font-mono">{file.name}</p>
                                            <p className="text-[#d8cbba] text-xs font-mono">{(file.size / 1024).toFixed(1)} KB (Ready for evaluation)</p>
                                        </div>
                                    ) : (
                                        <div className="space-y-2">
                                            <div className="w-12 h-12 bg-[#2a170e] border border-[#522a0e] rounded-full flex items-center justify-center mx-auto mb-2 group-hover:scale-110 transition-transform">
                                                <Upload className="w-6 h-6 text-[#f7bf47]" />
                                            </div>
                                            <p className="text-[#fff7e6] font-bold text-base">Drag & drop your .onnx model here</p>
                                            <p className="text-[#a89682] text-xs font-mono">or click to browse local files (max 5 MB)</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Verification Status Message */}
                        {verificationStatus !== 'idle' && (
                            <div className={`p-4 rounded text-xs font-mono border ${
                                verificationStatus === 'valid'
                                    ? 'bg-[#a3e635]/15 border-[#a3e635]/40 text-[#a3e635]'
                                    : verificationStatus === 'invalid'
                                    ? 'bg-[#f87171]/15 border-[#f87171]/40 text-[#f87171]'
                                    : 'bg-[#180e07] border-[#522a0e] text-[#d8cbba]'
                            }`}>
                                {verificationStatus === 'checking' && <span className="animate-pulse">⏳ </span>}
                                {verificationMsg}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={status === 'uploading' || status === 'evaluating' || verificationStatus !== 'valid'}
                            className={`w-full py-4 text-xs font-bold uppercase tracking-wider rounded font-[family-name:var(--font-pixel)] transition-all ${
                                (status === 'idle' && verificationStatus === 'valid') || status === 'error'
                                    ? 'stardew-btn-gold text-[#24140b] cursor-pointer'
                                    : 'bg-[#2a170e] border border-[#522a0e] text-[#6a5441] cursor-not-allowed opacity-50'
                            }`}
                        >
                            {status === 'idle' && '🚀 Launch Official Evaluation'}
                            {status === 'uploading' && 'Uploading Model...'}
                            {status === 'evaluating' && 'Running 25-Fish Simulation...'}
                            {status === 'success' && 'Evaluation Complete!'}
                            {status === 'error' && 'Retry Submission'}
                        </button>
                    </form>

                    {/* Progress during evaluation */}
                    {/* Progress during evaluation */}
                    {status === 'evaluating' && (
                        <div className="mt-8 stardew-box p-6 space-y-3">
                            <div className="flex items-center justify-between text-xs font-bold font-mono">
                                <span className="text-[#d8cbba]">Simulating: {evalProgress.fishName}</span>
                                <span className="text-[#f7bf47]">{evalProgress.current}/{evalProgress.total}</span>
                            </div>
                            <div className="w-full bg-[#180e07] border border-[#522a0e] rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-[#e89c25] to-[#f7bf47] h-full transition-all duration-300"
                                    style={{ width: `${evalProgress.total > 0 ? (evalProgress.current / evalProgress.total) * 100 : 0}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {status === 'success' && evalResult && (
                        <div className="mt-8 stardew-box p-6 space-y-4 border-2 border-[#a3e635]">
                            <div className="flex items-center gap-3">
                                <CheckCircle className="text-[#a3e635] w-7 h-7" />
                                <div>
                                    <h4 className="font-bold text-[#a3e635] font-[family-name:var(--font-pixel)] text-sm">
                                        Official Evaluation Complete
                                    </h4>
                                    <span className="text-[#d8cbba] text-xs font-mono">
                                        All {evalResult.nFish} fish × {evalResult.seedsPerFish} seeds (runSeed {evalResult.runSeed})
                                    </span>
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-3 text-center pt-2">
                                <div className="stardew-slot p-3">
                                    <div className="text-2xl font-bold font-[family-name:var(--font-pixel)] text-[#fff7e6]">
                                        {(evalResult.score * 100).toFixed(1)}
                                    </div>
                                    <div className="text-[10px] text-[#a89682] uppercase font-mono mt-1">Weighted Score</div>
                                </div>
                                <div className="stardew-slot p-3">
                                    <div className="text-2xl font-bold font-[family-name:var(--font-pixel)] text-[#a3e635]">
                                        {(evalResult.catchRate * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-[10px] text-[#a89682] uppercase font-mono mt-1">Catch Rate</div>
                                </div>
                                <div className="stardew-slot p-3">
                                    <div className="text-2xl font-bold font-[family-name:var(--font-pixel)] text-[#f7bf47]">
                                        {(evalResult.scoreHard * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-[10px] text-[#a89682] uppercase font-mono mt-1">Hard Fish Rate</div>
                                </div>
                            </div>

                            <p className="text-xs text-[#a89682] text-center italic font-mono pt-2">
                                Your result has been submitted to the challenge board.
                            </p>
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="mt-8 stardew-box p-6 border-2 border-[#f87171] flex items-center gap-4">
                            <AlertCircle className="text-[#f87171] w-7 h-7 shrink-0" />
                            <div>
                                <h4 className="font-bold text-[#f87171] font-[family-name:var(--font-pixel)] text-sm mb-1">
                                    Submission Failed
                                </h4>
                                <p className="text-[#fca5a5] text-xs leading-relaxed">{errorMsg}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <Footer />
        </main>
    );
}
