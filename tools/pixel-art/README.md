# Pixel-art sprites

Turns a generated sprite sheet into 24x24 transparent PNGs. The sheet itself is
not kept in the repo.

Only two sprites from the first sheet are used by the app and kept in
`src/assets/pixel/`: `icon-flame.png` and `icon-star.png`. The other names in
`sprites.json` (thirteen machine-part icons and a coin) were tried and dropped,
so a rerun should write somewhere else first (`--out some/scratch/dir`) and only
the sprites that are wanted copied over.

## Rerun on a new sheet

```sh
uv run --with pillow --with numpy python tools/pixel-art/process.py path/to/sheet.jpg
uv run --with pillow --with numpy python tools/pixel-art/contact.py contact-dark.png --background "#2e2e2e"
uv run --with pillow --with numpy python tools/pixel-art/contact.py contact-light.png --background "#f7f7f8"
```

The sheet must be a regular grid of icons on flat magenta (`#FF00FF`), one
icon per cell, in the order listed in `sprites.json` (row by row). Ask the
image model for very simple icons drawn on a 24x24 grid with huge square
pixels, a one-block charcoal outline, at most five flat colors each, no
gradients, glow, text, or grid lines.

Useful options for `process.py`: `--only NAME` (repeatable) to redo one
sprite, `--pitch N` to force the source pixel size when the estimate is off,
`--canvas` and `--margin` to change the output size, `--no-touchups`.

## What the script does

1. Keys the background by hue (a narrow band around magenta that is also
   saturated and bright), so pink and purple parts survive.
2. Slices the sheet into the grid from `sprites.json` and takes the bounding
   box of what is left in each cell.
3. Snaps every source pixel to the sprite's slice of the shared palette (the
   neutrals plus the colors listed for that sprite).
4. Estimates the size of one source "pixel" from where color edges fall, then
   gives each block its most common palette color. No averaging, so edges
   stay crisp and alpha is only ever 0 or 255.
5. Removes orphan pixels, turns every pixel on the silhouette edge into the
   outline color, and places the sprite on the canvas, centered, on a common
   baseline (`--margin` pixels from the bottom).
6. Applies `touchups.json` and copies `palette.json` next to the sprites.

## Config files

- `palette.json`: the shared palette, name to hex. Edit it here; the script
  copies it to `src/assets/pixel/palette.json` for the UI.
- `sprites.json`: grid size, the neutral colors every sprite may use, and for
  each cell the output name and its accent colors.
- `handdrawn.json`: sprites drawn by hand as character grids (`legend` maps
  each character to a palette name, or `null` for transparent). A name listed
  here ignores its sheet cell. The slab stack is drawn this way.
- `touchups.json`: per sprite, a list of `[x, y, color]` fixes in canvas
  coordinates, applied last (`null` clears a pixel). These depend on the
  sheet, so clear them when you switch to a new one.

## Palette

| Name | Hex | Used for |
| --- | --- | --- |
| outline | `#1c1c22` | every outline |
| steel-dark | `#4b4f5c` | steel shadow |
| steel | `#8a8f9c` | steel body |
| steel-light | `#c8ccd6` | steel highlight |
| paper | `#f4ecd8` | off-white |
| red | `#d63c3c` | sizing; flame |
| orange | `#ec6f2a` | tokens, around the model; flame |
| amber | `#f0a422` | numerics; shadow of yellow |
| yellow | `#f8d034` | speculative, capstone, star, coin |
| cream | `#fff09a` | highlight of yellow |
| green | `#2fa24c` | GPU |
| lime | `#96d02c` | structured output; highlight of green |
| lime-light | `#d0f470` | highlight of lime |
| teal | `#1fa596` | KV cache |
| teal-light | `#70e0cc` | highlight of teal |
| blue | `#3a74d8` | forward pass |
| cyan | `#2cb6e2` | serving; highlight of blue |
| cyan-light | `#90e2fc` | highlight of cyan |
| purple | `#8656d6` | quantization |
| purple-light | `#b99cff` | highlight of purple |
| pink | `#e26a94` | sampling |
| pink-light | `#ffb2c8` | highlight of pink |

The warm colors form one ramp (red, orange, amber, yellow, cream), so each
accent's highlight is the next color along.
