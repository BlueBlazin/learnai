"""Draw a labeled contact sheet of the sprites, scaled with nearest neighbor.

Run:
  uv run --with pillow --with numpy python tools/pixel-art/contact.py OUT.png --background "#2e2e2e"
"""

import argparse
import pathlib

from PIL import Image, ImageDraw

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--dir", default=str(REPO / "src" / "assets" / "pixel"))
    parser.add_argument("--background", default="#2e2e2e")
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    files = sorted(pathlib.Path(args.dir).glob("*.png"), key=lambda p: (not p.name.startswith("stage"), p.name))
    sprites = [Image.open(f).convert("RGBA") for f in files]
    side = max(s.width for s in sprites) * args.scale
    pad, label = 24, 20
    rows = -(-len(sprites) // args.cols)
    sheet = Image.new("RGB", (args.cols * (side + pad) + pad, rows * (side + pad + label) + pad), args.background)
    draw = ImageDraw.Draw(sheet)
    dark = sum(sheet.getpixel((0, 0))) < 384
    for i, (file, sprite) in enumerate(zip(files, sprites)):
        x = pad + (i % args.cols) * (side + pad)
        y = pad + (i // args.cols) * (side + pad + label)
        big = sprite.resize((sprite.width * args.scale, sprite.height * args.scale), Image.NEAREST)
        sheet.paste(big, (x, y), big)
        draw.text((x, y + side + 4), file.name, fill="#d0d0d0" if dark else "#303030")
    sheet.save(args.out)


if __name__ == "__main__":
    main()
