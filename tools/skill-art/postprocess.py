"""Turn raw skill generations into the WebP files the app bundles.

Usage:
  uv run --with pillow python postprocess.py RAW_DIR [--contact contact.jpg] [slug ...]

For each skill in skills.json (or only the named ones) this reads
RAW_DIR/<slug>.jpg, trims the off-white poster margin the model sometimes bakes
in, center-crops to a square, resizes, and writes src/assets/skills/<slug>.webp.
With --contact it also writes a labeled contact sheet of what it processed.
"""

import argparse
import json
import pathlib

from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / "src" / "assets" / "skills"
SIZE = 768
QUALITY = 86
PAPER = (241, 237, 224)
CHARACTER = "wanderer"


def is_paper(pixel: tuple[int, int, int]) -> bool:
    return all(abs(pixel[i] - PAPER[i]) <= 22 for i in range(3))


def margin(image: Image.Image) -> int:
    """Width of a uniform paper border, 0 when the art bleeds to the edges.
    A border counts only when all four edge strips are paper."""
    width, height = image.size
    pixels = image.load()

    def strip_is_paper(offset: int) -> bool:
        samples = []
        for step in range(0, width, 16):
            samples += [pixels[step, offset], pixels[step, height - 1 - offset]]
        for step in range(0, height, 16):
            samples += [pixels[offset, step], pixels[width - 1 - offset, step]]
        return sum(is_paper(sample) for sample in samples) >= 0.97 * len(samples)

    depth = 0
    while depth < width // 6 and strip_is_paper(depth):
        depth += 1
    return depth + 3 if depth >= 6 else 0


def finish(raw: Image.Image) -> Image.Image:
    image = raw.convert("RGB")
    trim = margin(image)
    if trim:
        image = image.crop((trim, trim, image.width - trim, image.height - trim))
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=pathlib.Path)
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--contact", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    skills = [skill for skill in json.loads((HERE / "skills.json").read_text()) if skill["slug"] != CHARACTER]
    if args.slugs:
        skills = [skill for skill in skills if skill["slug"] in args.slugs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for skill in skills:
        raw = args.raw_dir / f"{skill['slug']}.jpg"
        if not raw.exists():
            print("missing", raw.name)
            continue
        image = finish(Image.open(raw))
        image.save(args.out_dir / f"{skill['slug']}.webp", "WEBP", quality=QUALITY, method=6)
        done.append((skill, image))
    print("wrote", len(done), "files to", args.out_dir)

    if args.contact and done:
        cell, pad, label = 320, 16, 28
        columns = 4
        rows = -(-len(done) // columns)
        sheet = Image.new("RGB", (columns * (cell + pad) + pad, rows * (cell + label + pad) + pad), (46, 46, 46))
        draw = ImageDraw.Draw(sheet)
        for index, (skill, image) in enumerate(done):
            x = pad + (index % columns) * (cell + pad)
            y = pad + (index // columns) * (cell + label + pad)
            sheet.paste(image.resize((cell, cell), Image.LANCZOS), (x, y))
            draw.text((x, y + cell + 6), skill["label"], fill=(235, 235, 235))
        sheet.save(args.contact, quality=88)


if __name__ == "__main__":
    main()
