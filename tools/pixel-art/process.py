"""Turn a generated sprite sheet into small, palette-snapped pixel-art PNGs.

Steps: key out the magenta background, slice the sheet into grid cells, find
each object's bounding box, estimate the size of the source "pixels", take the
dominant palette color of every block, place the result on a square canvas on
a shared baseline, repair outlines, and apply hand touch-ups.

Run:
  uv run --with pillow --with numpy python tools/pixel-art/process.py SHEET.jpg

See README.md for the options and the config files.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb):
    """sRGB (0..255, shape [..., 3]) to CIE Lab."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def background_mask(rgb):
    """True where a pixel is the magenta key color.

    Hue-aware, so pink and purple accents survive: the key is a narrow hue
    band around 300 degrees that is also saturated and bright.
    """
    c = rgb.astype(np.float64) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    high, low = c.max(-1), c.min(-1)
    delta = high - low
    safe = np.where(delta == 0, 1, delta)
    hue = np.where(high == r, ((g - b) / safe) % 6, np.where(high == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60
    sat = np.where(high == 0, 0, delta / np.where(high == 0, 1, high))
    return (np.abs(hue - 300) < 24) & (sat > 0.62) & (high > 0.62)


def estimate_pitch(rgb, low=4.0, high=16.0):
    """Estimate the side of one source "pixel" in image pixels.

    Color edges in an upscaled pixel-art image fall on a regular grid. For each
    candidate pitch, measure how much of the edge energy lands on grid lines
    (best phase), on both axes. Finer grids that divide the true pitch score
    just as well, so take the largest pitch that scores close to the best.
    Returns (pitch, x_phase, y_phase).
    """
    gray = rgb.astype(np.float64)
    profile_x = np.abs(np.diff(gray, axis=1)).sum(axis=(0, 2))
    profile_y = np.abs(np.diff(gray, axis=0)).sum(axis=(1, 2))

    def score(profile, pitch):
        positions = np.arange(len(profile))
        best, best_phase = -1.0, 0.0
        for phase in np.arange(0, pitch, 0.5):
            distance = np.abs(((positions - phase + pitch / 2) % pitch) - pitch / 2)
            on_grid = profile[distance <= 1.0].sum()
            if on_grid > best:
                best, best_phase = on_grid, phase
        # Fraction of edge energy on the grid, relative to the fraction of
        # positions the grid lines cover (so a coarse grid is not penalized).
        covered = min(1.0, 3.0 / pitch)
        return best / max(profile.sum(), 1e-9) / covered, best_phase

    results = []
    for pitch in np.arange(low, high + 0.01, 0.1):
        sx, px = score(profile_x, pitch)
        sy, py = score(profile_y, pitch)
        results.append((sx * sy, pitch, px, py))
    top = max(r[0] for r in results)
    good = [r for r in results if r[0] >= 0.85 * top]
    _, pitch, px, py = max(good, key=lambda r: r[1])
    return pitch, px, py


def blocks_along(start, stop, pitch, phase):
    """Block edges covering [start, stop) on a grid of the given pitch/phase."""
    first = np.floor((start - phase) / pitch + 0.35)
    edges = [phase + first * pitch]
    while edges[-1] + pitch * 0.35 < stop:
        edges.append(edges[-1] + pitch)
    return edges


def downsample(index, pitch, px, py, max_cells):
    """Reduce an index image (-1 = transparent) to one cell per source pixel.

    Every cell takes the most common value in the middle of its block, which
    keeps edges crisp where averaging would blur them. If the object is bigger
    than max_cells, fall back to an evenly spaced grid of max_cells.
    """
    height, width = index.shape
    xs = blocks_along(0, width, pitch, px % pitch)
    ys = blocks_along(0, height, pitch, py % pitch)
    if max(len(xs), len(ys)) - 1 > max_cells:
        step = max(width, height) / max_cells
        xs = list(np.arange(0, width + step * 0.5, step))
        ys = list(np.arange(0, height + step * 0.5, step))
    out = np.full((len(ys) - 1, len(xs) - 1), -1, dtype=np.int64)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            inset_y = (ys[j + 1] - ys[j]) * 0.2
            inset_x = (xs[i + 1] - xs[i]) * 0.2
            y0, y1 = int(round(ys[j] + inset_y)), int(round(ys[j + 1] - inset_y))
            x0, x1 = int(round(xs[i] + inset_x)), int(round(xs[i + 1] - inset_x))
            block = index[max(y0, 0) : max(y1, y0 + 1), max(x0, 0) : max(x1, x0 + 1)]
            if block.size == 0:
                continue
            values, counts = np.unique(block, return_counts=True)
            out[j, i] = values[counts.argmax()]
    return out


def trim(cells):
    rows = np.where((cells >= 0).any(axis=1))[0]
    cols = np.where((cells >= 0).any(axis=0))[0]
    return cells[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def clean(cells, outline):
    """Remove orphan pixels, then make sure the silhouette is outlined."""
    padded = np.pad(cells, 1, constant_values=-1)
    opaque = padded >= 0
    neighbors = sum(
        np.roll(np.roll(opaque, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
    )
    padded[opaque & (neighbors == 0)] = -1
    opaque = padded >= 0
    touches_clear = np.zeros_like(opaque)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        touches_clear |= ~np.roll(np.roll(opaque, dy, 0), dx, 1)
    padded[opaque & touches_clear] = outline
    return padded[1:-1, 1:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet", help="generated sprite sheet on a magenta background")
    parser.add_argument("--config", default=str(HERE / "sprites.json"))
    parser.add_argument("--palette", default=str(HERE / "palette.json"))
    parser.add_argument("--touchups", default=str(HERE / "touchups.json"))
    parser.add_argument("--handdrawn", default=str(HERE / "handdrawn.json"))
    parser.add_argument("--out", default=str(REPO / "src" / "assets" / "pixel"))
    parser.add_argument("--canvas", type=int, default=32)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--pitch", type=float, default=0, help="source pixel size; 0 estimates it per sprite")
    parser.add_argument("--only", action="append", default=[], help="process only this sprite name (repeatable)")
    parser.add_argument("--no-touchups", action="store_true")
    args = parser.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text())
    palette = json.loads(pathlib.Path(args.palette).read_text())
    touchups = json.loads(pathlib.Path(args.touchups).read_text())
    handdrawn = json.loads(pathlib.Path(args.handdrawn).read_text())
    names = list(palette)
    colors = np.array([hex_to_rgb(palette[name]) for name in names], dtype=np.uint8)
    labs = rgb_to_lab(colors)
    outline = names.index("outline")

    sheet = np.asarray(Image.open(args.sheet).convert("RGB"))
    background = background_mask(sheet)
    cell_h = sheet.shape[0] / config["rows"]
    cell_w = sheet.shape[1] / config["cols"]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_cells = args.canvas - 2 * args.margin

    for position, sprite in enumerate(config["sprites"]):
        name = sprite["name"]
        if sprite.get("skip") or (args.only and name not in args.only):
            continue
        if name in handdrawn:
            # Drawn by hand as a character grid; the sheet cell is ignored.
            legend = handdrawn[name]["legend"]
            cells = np.array(
                [[-1 if legend[ch] is None else names.index(legend[ch]) for ch in line] for line in handdrawn[name]["rows"]]
            )
            pitch = 0.0
        else:
            row, col = divmod(position, config["cols"])
            y0, y1 = int(round(row * cell_h)), int(round((row + 1) * cell_h))
            x0, x1 = int(round(col * cell_w)), int(round((col + 1) * cell_w))
            mask = ~background[y0:y1, x0:x1]
            rows = np.where(mask.sum(axis=1) > 2)[0]
            cols = np.where(mask.sum(axis=0) > 2)[0]
            if len(rows) == 0:
                print(f"{name}: empty cell, skipped")
                continue
            top, bottom, left, right = y0 + rows[0], y0 + rows[-1] + 1, x0 + cols[0], x0 + cols[-1] + 1
            crop = sheet[top:bottom, left:right]
            crop_background = background[top:bottom, left:right]

            # Snap every source pixel to this sprite's slice of the shared palette.
            allowed = [names.index(n) for n in config["neutrals"] + sprite["colors"]]
            distance = np.linalg.norm(rgb_to_lab(crop)[:, :, None, :] - labs[allowed][None, None, :, :], axis=-1)
            index = np.array(allowed)[distance.argmin(axis=-1)]
            index[crop_background] = -1

            if args.pitch:
                pitch, px, py = args.pitch, 0.0, 0.0
            else:
                pitch, px, py = estimate_pitch(crop)
            cells = trim(downsample(index, pitch, px, py, max_cells))
            cells = trim(clean(cells, outline))
        height, width = cells.shape

        canvas = np.full((args.canvas, args.canvas), -1, dtype=np.int64)
        left_pad = (args.canvas - width) // 2
        top_pad = args.canvas - args.margin - height  # common baseline
        canvas[top_pad : top_pad + height, left_pad : left_pad + width] = cells

        if not args.no_touchups:
            for x, y, color in touchups.get(name, []):
                canvas[y, x] = -1 if color is None else names.index(color)

        rgba = np.zeros((args.canvas, args.canvas, 4), dtype=np.uint8)
        opaque = canvas >= 0
        rgba[opaque, :3] = colors[canvas[opaque]]
        rgba[opaque, 3] = 255
        Image.fromarray(rgba, "RGBA").save(out_dir / f"{name}.png")
        used = len(np.unique(canvas[opaque]))
        print(f"{name}: pitch {pitch:.1f}, {width}x{height} cells, {used} colors")

    shutil.copyfile(args.palette, out_dir / "palette.json")


if __name__ == "__main__":
    main()
