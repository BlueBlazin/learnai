"""Turn raw scene generations into the shipped WebP files.

Usage:
  uv run --with pillow python postprocess.py RAW_DIR [--contact contact.jpg]

For each scene in scenes.json this reads RAW_DIR/<slug>.jpg, trims the baked-in
poster margin listed in the scene's "crop" field ([left, top, right, bottom] in
source pixels, omitted when there is none), center-crops to the shared aspect,
resizes to the shared size, and writes src/assets/scenes/<slug>.webp. It then
measures the accent actually printed in each final and writes
src/assets/scenes/scenes.json for the UI. With --contact it also writes a
labeled contact sheet.
"""

import argparse
import json
import pathlib
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / "src" / "assets" / "scenes"
SIZE = (1584, 672)  # the model's native 21:9 output
QUALITY = 85


def finish(raw: Image.Image, crop: list[int] | None) -> Image.Image:
    image = raw.convert("RGB")
    left, top, right, bottom = crop or [0, 0, 0, 0]
    image = image.crop((left, top, image.width - right, image.height - bottom))
    target = SIZE[0] / SIZE[1]
    if image.width / image.height > target:
        width = round(image.height * target)
        offset = (image.width - width) // 2
        image = image.crop((offset, 0, offset + width, image.height))
    else:
        height = round(image.width / target)
        offset = (image.height - height) // 2
        image = image.crop((0, offset, image.width, offset + height))
    return image.resize(SIZE, Image.LANCZOS)


def measure_accent(image: Image.Image) -> str:
    """The dominant saturated color: the heaviest cluster when each pixel is
    weighted by its chroma squared, which favors the full accent over its tint,
    the muted steel blue, the paper, and the ink."""
    small = image.resize((396, 168), Image.BOX)
    weights: dict[tuple[int, int, int], float] = defaultdict(float)
    data = small.tobytes()
    pixels = [tuple(data[i : i + 3]) for i in range(0, len(data), 3)]
    for r, g, b in pixels:
        chroma = max(r, g, b) - min(r, g, b)
        if chroma < 70 or max(r, g, b) < 110:
            continue
        weights[(r >> 3, g >> 3, b >> 3)] += chroma * chroma
    best = max(weights, key=weights.get)
    center = [channel * 8 + 4 for channel in best]
    near = [p for p in pixels if all(abs(p[i] - center[i]) <= 14 for i in range(3))]
    mean = [round(sum(p[i] for p in near) / len(near)) for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*mean)


def contact_sheet(finals: list[tuple[str, Image.Image]], path: pathlib.Path) -> None:
    columns, width, gap, label = 3, 1800, 24, 30
    cell = (width - gap * (columns + 1)) // columns
    cell_height = round(cell * SIZE[1] / SIZE[0])
    rows = -(-len(finals) // columns)
    sheet = Image.new("RGB", (width, gap + rows * (cell_height + label + gap)), "#1f1f1f")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except OSError:
        font = ImageFont.load_default()
    for index, (text, image) in enumerate(finals):
        x = gap + (index % columns) * (cell + gap)
        y = gap + (index // columns) * (cell_height + label + gap)
        sheet.paste(image.resize((cell, cell_height), Image.LANCZOS), (x, y))
        draw.text((x, y + cell_height + 7), text, fill="#d8d8d8", font=font)
    sheet.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=pathlib.Path)
    parser.add_argument("--contact", type=pathlib.Path)
    args = parser.parse_args()

    scenes = json.loads((HERE / "scenes.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, finals = [], []
    for scene in scenes:
        slug = scene["slug"]
        image = finish(Image.open(args.raw_dir / f"{slug}.jpg"), scene.get("crop"))
        image.save(OUT_DIR / f"{slug}.webp", "WEBP", quality=QUALITY, method=6)
        accent = measure_accent(image)
        manifest.append({"slug": slug, "place": scene["place"], "accent": accent})
        finals.append((f"{scene['stage']:02d}  {slug}   {accent}", image))
        print(slug, "asked", scene["accent"]["hex"], "printed", accent)
    (OUT_DIR / "scenes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.contact:
        contact_sheet(finals, args.contact)


if __name__ == "__main__":
    main()
