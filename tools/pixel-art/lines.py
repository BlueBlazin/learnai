"""Derive the line mask of each stage sprite: `stage-XX-name.lines.png`.

A line mask is opaque where the finished sprite has the outline color (the
silhouette edge plus the interior line work) and transparent elsewhere. The UI
uses it as a CSS mask to draw a part that is not built yet as a line drawing
in its stage color. It is derived from the finished PNGs, so it never needs
the source sheet.

    uv run --with pillow --with numpy python tools/pixel-art/lines.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SPRITES = HERE.parent.parent / "src" / "assets" / "pixel"
SUFFIX = ".lines.png"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def line_mask(sprite: Image.Image, outline: tuple[int, int, int]) -> Image.Image:
    pixels = np.asarray(sprite.convert("RGBA"))
    is_line = (pixels[..., 3] == 255) & (pixels[..., :3] == np.array(outline, dtype=np.uint8)).all(axis=-1)
    out = np.zeros_like(pixels)
    out[is_line] = (*outline, 255)
    return Image.fromarray(out, "RGBA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sprites", type=Path, default=SPRITES, help="directory holding the finished PNGs")
    parser.add_argument("--glob", default="stage-*.png", help="which sprites get a line mask")
    args = parser.parse_args()

    outline = hex_to_rgb(json.loads((HERE / "palette.json").read_text())["outline"])
    sources = sorted(path for path in args.sprites.glob(args.glob) if not path.name.endswith(SUFFIX))
    for source in sources:
        mask = line_mask(Image.open(source), outline)
        target = source.with_name(source.stem + SUFFIX)
        mask.save(target, optimize=True)
        count = int((np.asarray(mask)[..., 3] > 0).sum())
        print(f"{target.name}: {count} line pixels")


if __name__ == "__main__":
    main()
