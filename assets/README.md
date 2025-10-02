# Stardew Valley Fishing Game Assets

This directory contains individual sprite assets extracted from the Stardew Valley spritesheet (`Cursors.png`) for use in the fishing minigame.

## Sprites

| File | Description | Dimensions | Usage |
|------|-------------|------------|-------|
| `background.png` | Bobber bar background | 37×150 | Main UI background |
| `fish_normal.png` | Normal fish icon | 20×20 | Regular fish indicator |
| `fish_boss.png` | Boss fish icon | 20×20 | Legendary fish indicator |
| `catch_bar_top.png` | Catch area top edge | 9×2 | Green catch zone top |
| `catch_bar_mid.png` | Catch area middle | 9×1 | Green catch zone body |
| `catch_bar_bot.png` | Catch area bottom edge | 9×2 | Green catch zone bottom |
| `handle.png` | Bobber handle | 5×10 | UI handle element |

## Usage

These sprites are automatically loaded by `environment.py` and scaled 4× to match Stardew Valley's authentic UI scaling. The sprites maintain the original pixel-perfect quality from the game.

## Generation

These assets were extracted using the `crop_sprites.py` utility from the original Stardew Valley `Cursors.png` spritesheet.

## Scaling

All sprites are displayed at 4× their original size to match Stardew Valley's UI scale:
- Original size → Display size
- 37×150 → 148×600 (background)
- 20×20 → 80×80 (fish icons)
- 9×2 → 36×8 (catch bar edges)
- etc.