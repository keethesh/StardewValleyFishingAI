# Cinematic Training Visualization

Modern, animated visualization of AI training progress with milestone highlights and smooth transitions.

## Features

- **Milestone Timeline**: Horizontal timeline showing all training milestones with category-coded markers
- **Win Rate Progression**: Smooth area chart showing learning curve from 0% → 99%
- **Behavior Comparison**: Multi-line chart comparing success rates across all 5 fish behaviors
- **Animated Replay**: 30-second cinematic replay with automatic slow-motion on milestones
- **Milestone Popups**: Auto-appearing cards with spring animations highlighting key achievements
- **Playback Controls**: Play/pause and speed control (0.5x, 1x, 2x)

## Usage

### 1. Export Training Data

```bash
python export_training_data.py
```

This converts your latest CSV training logs to `visualization/training_data.json`

### 2. Open Visualization

Simply open `visualization/index.html` in your browser (Chrome/Firefox recommended)

Or use a local server:
```bash
cd visualization
python -m http.server 8000
# Then visit: http://localhost:8000
```

### 3. Screen Record for Video

- Click Play button
- Use OBS or screen recording software
- Export as MP4 for YouTube

## Customization

### Change Theme

Edit CSS variables in `styles.css`:

```css
:root {
    /* Colors */
    --bg-primary: #0a0a0a;        /* Background */
    --accent-primary: #007aff;     /* Primary accent */

    /* Easy theme swaps: */
    /* Cyberpunk: --bg-primary: #000; --accent-primary: #0ff; */
    /* Stardew: --bg-primary: #1a1a1a; --accent-primary: #6ab04c; */
}
```

### Adjust Animation Speed

Edit constants in `animation.js`:

```javascript
const ANIMATION_DURATION = 30;        // Total seconds (default: 30)
const MILESTONE_SLOWDOWN = 0.3;       // Slow-mo speed (0.3 = 30%)
const MILESTONE_PAUSE = 0.5;          // Pause duration at milestones
```

### Add Custom Visuals

The modular structure makes it easy to add charts:

1. Add HTML element in `index.html`
2. Create D3 visualization in `animation.js`
3. Style in `styles.css` using existing CSS variables

## File Structure

```
visualization/
├── index.html           # Main HTML structure
├── styles.css          # Modular theme system with CSS variables
├── animation.js        # D3.js + GSAP animation logic
├── training_data.json  # Exported training metrics (auto-generated)
└── README.md          # This file
```

## Dependencies

All loaded via CDN (no installation needed):
- D3.js v7 (data visualization)
- GSAP 3.12 (smooth animations)

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (may need --webkit- prefixes)

## Performance

Optimized for smooth 60fps animation:
- Hardware-accelerated CSS transforms
- Efficient D3 rendering
- GSAP performance optimizations
- Tested with 1500+ episodes

## Tips for Best Results

1. **Recording**: Use 1080p resolution at 60fps
2. **Playback**: Start with 1x speed to show milestone details, then 2x for final stretch
3. **Editing**: Slow-mo the most epic milestones (100-streak, 99% win rate) in your video editor
4. **Audio**: Add background music in post-production
