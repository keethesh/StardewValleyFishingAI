'use client';

import { useId } from 'react';

// Fixed seed pseudo-random positions for crisp pixel stars
const STARS = [
  { x: 5, y: 8, size: 2, delay: 0, opacity: 0.9, color: '#ffffff' },
  { x: 12, y: 22, size: 3, delay: 1.2, opacity: 0.8, color: '#ffe699' },
  { x: 19, y: 14, size: 2, delay: 2.5, opacity: 0.6, color: '#b8d4ff' },
  { x: 27, y: 32, size: 2, delay: 0.7, opacity: 0.85, color: '#ffffff' },
  { x: 34, y: 9, size: 3, delay: 3.1, opacity: 0.95, color: '#ffd67a' },
  { x: 42, y: 26, size: 2, delay: 1.8, opacity: 0.7, color: '#ffffff' },
  { x: 48, y: 16, size: 3, delay: 0.4, opacity: 0.85, color: '#c2ddff' },
  { x: 55, y: 38, size: 2, delay: 2.2, opacity: 0.65, color: '#ffffff' },
  { x: 62, y: 12, size: 2, delay: 1.5, opacity: 0.9, color: '#ffe699' },
  { x: 71, y: 28, size: 3, delay: 2.9, opacity: 0.8, color: '#ffffff' },
  { x: 78, y: 15, size: 2, delay: 0.2, opacity: 0.75, color: '#b8d4ff' },
  { x: 84, y: 35, size: 2, delay: 3.6, opacity: 0.6, color: '#ffffff' },
  { x: 91, y: 18, size: 3, delay: 1.1, opacity: 0.9, color: '#ffd67a' },
  { x: 96, y: 27, size: 2, delay: 2.7, opacity: 0.85, color: '#ffffff' },
  // Second cluster for taller screens
  { x: 8, y: 48, size: 2, delay: 1.7, opacity: 0.6, color: '#ffffff' },
  { x: 23, y: 55, size: 2, delay: 0.9, opacity: 0.7, color: '#ffe699' },
  { x: 38, y: 62, size: 3, delay: 2.3, opacity: 0.8, color: '#b8d4ff' },
  { x: 67, y: 52, size: 2, delay: 1.4, opacity: 0.65, color: '#ffffff' },
  { x: 82, y: 58, size: 2, delay: 3.3, opacity: 0.75, color: '#ffd67a' },
  { x: 93, y: 64, size: 2, delay: 0.6, opacity: 0.6, color: '#ffffff' },
];

// Floating golden fireflies
const FIREFLIES = [
  { left: '15%', top: '35%', tx: 30, ty: -40, duration: 9, delay: 0 },
  { left: '40%', top: '60%', tx: -45, ty: -55, duration: 12, delay: 2 },
  { left: '72%', top: '40%', tx: 35, ty: -45, duration: 10, delay: 4 },
  { left: '88%', top: '70%', tx: -30, ty: -35, duration: 11, delay: 1 },
  { left: '28%', top: '80%', tx: 40, ty: -50, duration: 13, delay: 3 },
];

export default function TwilightBackground() {
  const id = useId();

  return (
    <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden select-none">
      {/* Deep Royal Twilight Indigo Gradient */}
      <div 
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(180deg, #111526 0%, #171e36 28%, #1f2747 58%, #27223b 82%, #121a28 100%)',
        }}
      />

      {/* Atmospheric Radial Starlight & Lantern Glow */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
            radial-gradient(ellipse 90% 60% at 50% -10%, rgba(66, 92, 160, 0.35) 0%, transparent 70%),
            radial-gradient(ellipse 60% 45% at 85% 25%, rgba(247, 191, 71, 0.10) 0%, transparent 65%),
            radial-gradient(ellipse 80% 50% at 15% 75%, rgba(41, 74, 110, 0.30) 0%, transparent 70%),
            radial-gradient(ellipse 70% 40% at 50% 105%, rgba(18, 30, 48, 0.85) 0%, transparent 80%)
          `,
        }}
      />

      {/* Subtle Stardew Pixel Grid Texture Overlay */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage: `
            linear-gradient(to right, #ffffff 1px, transparent 1px),
            linear-gradient(to bottom, #ffffff 1px, transparent 1px)
          `,
          backgroundSize: '32px 32px',
        }}
      />

      {/* Crisp Pixel Stars */}
      <div className="absolute inset-0">
        {STARS.map((star, i) => (
          <span
            key={`${id}-star-${i}`}
            className="absolute rounded-none transform-gpu animate-twinkle"
            style={{
              left: `${star.x}%`,
              top: `${star.y}%`,
              width: `${star.size}px`,
              height: `${star.size}px`,
              backgroundColor: star.color,
              boxShadow: `0 0 ${star.size * 2}px ${star.color}`,
              opacity: star.opacity,
              animationDelay: `${star.delay}s`,
            }}
          />
        ))}
      </div>

      {/* Floating Stardew Fireflies */}
      <div className="absolute inset-0">
        {FIREFLIES.map((ff, i) => (
          <span
            key={`${id}-ff-${i}`}
            className="absolute rounded-full pointer-events-none animate-firefly"
            style={{
              left: ff.left,
              top: ff.top,
              width: '4px',
              height: '4px',
              backgroundColor: '#ffd659',
              boxShadow: '0 0 8px 3px rgba(255, 214, 89, 0.75), 0 0 16px 6px rgba(255, 186, 43, 0.35)',
              animationDuration: `${ff.duration}s`,
              animationDelay: `${ff.delay}s`,
              ['--tx' as string]: `${ff.tx}px`,
              ['--ty' as string]: `${ff.ty}px`,
            }}
          />
        ))}
      </div>

      {/* Subtle Night Pond Water Ripple Line at Bottom */}
      <div 
        className="absolute bottom-0 left-0 right-0 h-16 opacity-35"
        style={{
          background: 'repeating-linear-gradient(180deg, transparent 0px, transparent 4px, rgba(93, 165, 219, 0.15) 5px, transparent 6px)',
        }}
      />
    </div>
  );
}
