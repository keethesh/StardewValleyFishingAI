
'use client';

import { useEffect, useRef, useState } from 'react';
import { FishingGame, PHYSICS, Fish } from '@/lib/game-logic';
import { InferenceSession, Tensor } from 'onnxruntime-web';

interface Props {
    fish: Fish;
    modelUrl?: string; // If provided, AI plays
    width?: number;
    height?: number;
    onFinish?: (success: boolean) => void;
}

export default function FishingGameComponent({ fish, modelUrl, width = 200, height = 600, onFinish }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [game, setGame] = useState<FishingGame | null>(null);
    const [score, setScore] = useState(0);
    const [session, setSession] = useState<InferenceSession | null>(null);
    const requestRef = useRef<number>(0);
    const [isAI, setIsAI] = useState(false);
    const [controls, setControls] = useState({ pressing: false });
    const [message, setMessage] = useState('');

    // Load ONNX model if provided
    useEffect(() => {
        if (modelUrl) {
            setIsAI(true);
            const loadModel = async () => {
                try {
                    const sess = await InferenceSession.create(modelUrl, {
                        executionProviders: ['webgl', 'wasm'],
                        graphOptimizationLevel: 'all'
                    });
                    setSession(sess);
                } catch (e) {
                    console.error("Failed to load model", e);
                    setMessage("Failed to load AI");
                }
            };
            loadModel();
        } else {
            setIsAI(false);
            setSession(null);
        }
    }, [modelUrl]);

    // Initialize Game
    useEffect(() => {
        const newGame = new FishingGame(fish);
        setGame(newGame);
        setMessage('');
        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        }
    }, [fish]);


    const [images, setImages] = useState<Record<string, HTMLImageElement>>({});

    // Load Sprites
    useEffect(() => {
        const spriteNames = [
            'background', 'catch_bar_top', 'catch_bar_mid', 'catch_bar_bot',
            'fish_normal', 'fish_boss'
        ];
        const loaded: Record<string, HTMLImageElement> = {};
        let count = 0;

        spriteNames.forEach(name => {
            const img = new Image();
            img.src = `/sprites/${name}.png`;
            img.onload = () => {
                loaded[name] = img;
                count++;
                if (count === spriteNames.length) {
                    setImages(loaded);
                }
            };
        });
    }, []);

    // Main Game Loop
    useEffect(() => {
        if (!game || (isAI && !session)) return;

        let lastFrameTime = 0;
        const TARGET_FPS = 60;
        const FPS_INTERVAL = 1000 / TARGET_FPS;

        const loop = async (timestamp: number) => {
            if (!lastFrameTime) lastFrameTime = timestamp;
            const elapsed = timestamp - lastFrameTime;

            if (elapsed >= FPS_INTERVAL) {
                // Adjust for next frame, allowing for some drift catch-up but capping to prevent spiral
                lastFrameTime = timestamp - (elapsed % FPS_INTERVAL);

                let action: 0 | 1 = 0;

                if (!game.state.done) {
                    if (isAI && session) {
                        const inputData = game.getObservation();
                        const tensor = new Tensor('float32', inputData, [1, 8]);
                        try {
                            const results = await session.run({ state: tensor });
                            const output = results.q_values.data as Float32Array;
                            action = output[0] > output[1] ? 0 : 1;
                        } catch (e) {
                            console.error("Inference error", e);
                        }
                    } else {
                        action = controls.pressing ? 1 : 0;
                    }

                    const result = game.step(action);

                    if (result.done) {
                        setMessage(result.state.success ? 'CAUGHT!' : 'LOST!');
                        if (onFinish) onFinish(result.state.success);
                    }
                }
            }

            draw(game);
            requestRef.current = requestAnimationFrame(loop);
        };

        requestRef.current = requestAnimationFrame(loop);
        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        };
    }, [game, session, isAI, controls.pressing, images]);

    // Inputs
    const handleDown = () => setControls({ pressing: true });
    const handleUp = () => setControls({ pressing: false });

    // Keyboard Spacebar listener for desktop players
    useEffect(() => {
        if (isAI) return;
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.code === 'Space') {
                e.preventDefault();
                setControls({ pressing: true });
            }
        };
        const onKeyUp = (e: KeyboardEvent) => {
            if (e.code === 'Space') {
                e.preventDefault();
                setControls({ pressing: false });
            }
        };
        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
        };
    }, [isAI]);
    // Drawing
    const draw = (g: FishingGame) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.imageSmoothingEnabled = false; // Pixel art look
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, width, height);

        if (Object.keys(images).length < 6) return; // Wait for images

        const s = g.state;

        // Sprite Scale - x4 as requested
        const VISUAL_SCALE = 4.0;

        // Background
        const bg = images['background'];
        // Center background: The background is narrow (track), so we center it.
        const bgW = bg.width * VISUAL_SCALE;
        const bgH = bg.height * VISUAL_SCALE;
        const bgX = (width - bgW) / 2;
        const bgY = (height - bgH) / 2;

        // Draw Track Background
        ctx.drawImage(bg, bgX, bgY, bgW, bgH);

        // Physics to Canvas Mapping
        // Python: Track height is 568px internal units.
        // The background sprite track area is visually roughly 142px tall * 4 = 568px? 
        // Let's rely on the visual alignment. 
        // In the original sprite, the track top is roughly at pixel 12, height 140ish.
        // We'll map the physics (0-568) to the visual track area.

        // Visual calibration based on standard Stardew assets:
        // Track Top Offset: ~3% of height
        // Track Height: ~93% of height
        const trackTop = bgY + (25 * (VISUAL_SCALE / 2)); // approximated offset
        const trackHeight = bgH - (50 * (VISUAL_SCALE / 2));

        const physicsToCanvas = (val: number) => trackTop + (val / 568.0) * trackHeight;

        // --- Draw Bar ---
        const barPos = s.bobberBarPos;
        const barHeightPhysics = s.bobberBarHeight;

        const barY = physicsToCanvas(barPos);
        const barH = (barHeightPhysics / 568.0) * trackHeight;

        const top = images['catch_bar_top'];
        const mid = images['catch_bar_mid'];
        const bot = images['catch_bar_bot'];

        const barW = top.width * VISUAL_SCALE;
        const barX = (width - barW) / 2;

        // Top cap
        ctx.drawImage(top, barX, barY, barW, top.height * VISUAL_SCALE);

        // Middle section (stretched)
        const midY = barY + (top.height * VISUAL_SCALE);
        const botY = barY + barH - (bot.height * VISUAL_SCALE);
        const midH = botY - midY;

        if (midH > 0) {
            ctx.drawImage(mid, barX, midY, barW, midH);
        }

        // Bottom cap
        ctx.drawImage(bot, barX, botY, barW, bot.height * VISUAL_SCALE);

        // --- Draw Fish ---
        const fishImg = g.fish.difficulty >= 80 ? images['fish_boss'] : images['fish_normal'];
        const fishW = fishImg.width * VISUAL_SCALE;
        const fishH = fishImg.height * VISUAL_SCALE;
        const fishX = (width - fishW) / 2;
        const fishY = physicsToCanvas(s.bobberPosition) + (barH * 0.1); // slight visual offset to center on point

        ctx.drawImage(fishImg, fishX, fishY, fishW, fishH);

        // --- Progress Bar (Right side, connected to track) ---
        // Stardew has it attached to the right.
        const progW = 10 * VISUAL_SCALE; // thicker
        const progH = bgH - (20 * VISUAL_SCALE);
        const progX = bgX + bgW + 10; // offset from track
        const progY = bgY + (10 * VISUAL_SCALE);

        // Draw progress bg
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(progX, progY, progW, progH);

        // Border
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 4;
        ctx.strokeRect(progX, progY, progW, progH);

        // Fill
        const catchPct = s.distanceFromCatching; // 0 to 1
        const fillH = catchPct * progH;

        // Gradient Color
        const r = Math.min(255, Math.floor((2.0 - 2.0 * catchPct) * 255));
        const g_col = Math.min(255, Math.floor((2.0 * catchPct) * 255));

        ctx.fillStyle = `rgb(${r},${g_col},0)`;
        ctx.fillRect(progX, progY + progH - fillH, progW, fillH);

        // --- Text & Overlay ---
        ctx.fillStyle = 'white';
        ctx.font = 'bold 20px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${(catchPct * 100).toFixed(0)}%`, progX + (progW / 2), progY + progH + 30);
    };

    return (
        <div className="inline-flex flex-col items-center">
            <div className="relative inline-block stardew-box p-2 shadow-2xl">
                <canvas
                    ref={canvasRef}
                    width={width}
                    height={height}
                    onMouseDown={handleDown}
                    onMouseUp={handleUp}
                    onTouchStart={handleDown}
                    onTouchEnd={handleUp}
                    className="cursor-pointer active:cursor-grabbing touch-none block rounded-sm bg-[#120a06]"
                />
                {message && (
                    <div className="absolute inset-0 m-2 flex items-center justify-center bg-black/60 backdrop-blur-[2px] rounded-sm">
                        <div className={`text-2xl md:text-3xl font-bold font-[family-name:var(--font-pixel)] ${
                            message === 'CAUGHT!' ? 'text-[#a3e635]' : 'text-[#f87171]'
                        } animate-bounce drop-shadow-[0_4px_4px_rgba(0,0,0,0.8)]`}>
                            {message}
                        </div>
                    </div>
                )}
                {!isAI && !message && (
                    <div className="hidden md:block absolute bottom-4 left-0 right-0 text-center text-amber-100/60 text-xs font-mono pointer-events-none drop-shadow">
                        Hold Click or Spacebar
                    </div>
                )}
                {isAI && (
                    <div className="absolute top-4 right-4 text-[10px] font-bold font-[family-name:var(--font-pixel)] bg-[#6b3813] border border-[#b86e33] px-2 py-1 rounded text-amber-200 shadow">
                        AI PILOT
                    </div>
                )}
            </div>

            {/* Dedicated mobile tap target */}
            {!isAI && (
                <div className="w-full mt-3 md:hidden">
                    <button
                        type="button"
                        onMouseDown={handleDown}
                        onMouseUp={handleUp}
                        onTouchStart={handleDown}
                        onTouchEnd={handleUp}
                        className={`w-full py-3.5 px-4 text-xs font-bold font-[family-name:var(--font-pixel)] uppercase tracking-wider rounded-md select-none touch-none transition-all ${
                            controls.pressing
                                ? 'bg-[#c97816] text-[#24140b] translate-y-1 shadow-none'
                                : 'stardew-btn-gold text-[#24140b]'
                        }`}
                    >
                        {controls.pressing ? '⚡ REELING IN...' : '🎣 HOLD TO REEL'}
                    </button>
                </div>
            )}
        </div>
    );
}
