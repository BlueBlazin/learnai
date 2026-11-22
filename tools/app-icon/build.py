"""Build the app icon master (app-icon.png, 1024x1024) from art.jpg.

Usage:
  uv run --with pillow --with numpy python tools/app-icon/build.py
  pnpm tauri icon tools/app-icon/app-icon.png

Two steps. First the art's red sky is reprinted in cyan: each pixel is read as
a blend of the two measured inks it lies between and rebuilt from the
replacement inks, so the drawing and its edges do not change. Then the square
art is cut to the macOS icon shape (an 824 px body on a 1024 px canvas, with a
soft shadow), which is what `tauri icon` expects as its source.
"""

import itertools
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent

PAPER = "#efebdf"
# Inks measured from art.jpg, and what each is printed as in the icon.
MEASURED = {"sky": "#ce3331", "band": "#ef6050", "ink": "#0e1c25", "paper": PAPER, "steel": "#587c96"}
SKY = "#2cb6e2"  # cyan, the accent of the "village" scene
BAND_TINT = 0.35  # the horizon band is the sky ink mixed this far toward paper

CANVAS = 1024
BODY = 824


def rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], float)


def recolor(art: Image.Image) -> Image.Image:
    pixels = np.asarray(art.convert("RGB"), float)
    keys = list(MEASURED)
    old = np.stack([rgb(MEASURED[key]) for key in keys])
    printed = dict(MEASURED, sky=rgb(SKY), band=rgb(SKY) * (1 - BAND_TINT) + rgb(PAPER) * BAND_TINT)
    new = np.stack([printed[key] if isinstance(printed[key], np.ndarray) else rgb(printed[key]) for key in keys])

    shape = pixels.shape[:2]
    best = np.full(shape, np.inf)
    first = np.zeros(shape, int)
    second = np.zeros(shape, int)
    blend = np.zeros(shape)
    for i, j in itertools.combinations(range(len(keys)), 2):
        step = old[j] - old[i]
        t = np.clip(((pixels - old[i]) @ step) / (step @ step), 0, 1)
        distance = np.linalg.norm(pixels - (old[i] + t[..., None] * step), axis=2)
        closer = distance < best
        best[closer] = distance[closer]
        first[closer] = i
        second[closer] = j
        blend[closer] = t[closer]

    out = new[first] * (1 - blend[..., None]) + new[second] * blend[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype("uint8"))


def icon_shape(size: int, exponent: float = 5.0, oversample: int = 4) -> Image.Image:
    """A superellipse, close to the continuous-corner shape of macOS icons."""
    big = size * oversample
    mask = Image.new("L", (big, big), 0)
    half = big / 2
    points = []
    for step in range(720):
        angle = 2 * math.pi * step / 720
        c, s = math.cos(angle), math.sin(angle)
        points.append(
            (half + half * math.copysign(abs(c) ** (2 / exponent), c),
             half + half * math.copysign(abs(s) ** (2 / exponent), s))
        )
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask.resize((size, size), Image.LANCZOS)


def main() -> None:
    art = recolor(Image.open(HERE / "art.jpg")).resize((BODY, BODY), Image.LANCZOS)
    mask = icon_shape(BODY)
    pad = (CANVAS - BODY) // 2

    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 90), (pad, pad + 12), mask)
    icon = shadow.filter(ImageFilter.GaussianBlur(14))
    icon.paste(art, (pad, pad), mask)
    icon.save(HERE / "app-icon.png")
    print("wrote", HERE / "app-icon.png")


if __name__ == "__main__":
    main()
