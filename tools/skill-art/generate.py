"""Generate skill graphics from skills.json.

Usage:
  GEMINI_API_KEY=... python3 generate.py --out-dir RAW_DIR --style-ref SCENE_IMAGE \
      [--character-ref WANDERER_IMAGE] [--variants N] [slug ...]

With no slugs, every skill is generated (the `wanderer` character study only
when named). Each result is written to RAW_DIR/<slug>.jpg, or
RAW_DIR/<slug>-<n>.jpg with --variants. Keep RAW_DIR outside the repo.

The key is read from the GEMINI_API_KEY environment variable only, sent in a
header, and never printed. Standard library only.
"""

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import pathlib
import sys
import urllib.error
import urllib.request

MODEL = "gemini-nano-banana-2.1"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
HERE = pathlib.Path(__file__).resolve().parent
CHARACTER = "wanderer"

STYLE_ONLY = (
    "The attached image is a style reference only. Match its flat screenprint treatment, its handling "
    "of ink, its crisp cut-paper edges, and its level of simplification exactly, so this picture looks "
    "like another print from the same series. Do not copy its scene, its composition, its colors other "
    "than the shared inks, or its off-white margin."
)
STYLE_AND_CHARACTER = (
    "Two images are attached. The first is a style reference only. Match its flat screenprint "
    "treatment, its handling of ink, its crisp cut-paper edges, and its level of simplification "
    "exactly, so this picture looks like another print from the same series. Do not copy its scene, "
    "its composition, or its colors other than the shared inks. The second image shows the traveler "
    "who appears in this picture. Draw the same traveler: the same hat, coat, rucksack with its "
    "blanket roll, proportions, and inks, as the same faceless flat silhouette. Do not copy the "
    "second image's pose, viewing angle, landscape, or colors other than the shared inks; only the "
    "character and how it is rendered. The pose and the viewing angle come from the Scene text below, "
    "and the rucksack and scarf take this print's own accent."
)


def build_prompt(skill: dict, with_character: bool) -> str:
    template = (HERE / "style.txt").read_text()
    return template.format(
        refs=STYLE_AND_CHARACTER if with_character else STYLE_ONLY,
        scene=skill["scene"],
        accent_name=skill["accent"]["name"],
        accent_hex=skill["accent"]["hex"],
        inks=skill["inks"],
    )


def generate(prompt: str, refs: list[pathlib.Path], key: str) -> bytes:
    parts = []
    for ref in refs:
        mime = mimetypes.guess_type(ref.name)[0] or "image/jpeg"
        parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(ref.read_bytes()).decode()}})
    parts.append({"text": prompt})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"},
        },
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:600]}")
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("inlineData"):
                return base64.b64decode(part["inlineData"]["data"])
    raise RuntimeError("no image in response: " + json.dumps(payload)[:600])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*", help="skill slugs; every skill when omitted")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--style-ref", type=pathlib.Path, required=True, help="a finished scene")
    parser.add_argument("--character-ref", type=pathlib.Path, help="the approved wanderer study")
    parser.add_argument("--variants", type=int, default=1, help="takes per slug")
    parser.add_argument("--jobs", type=int, default=4, help="requests in flight at once")
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY in the environment")

    skills = {skill["slug"]: skill for skill in json.loads((HERE / "skills.json").read_text())}
    unknown = [slug for slug in args.slugs if slug not in skills]
    if unknown:
        raise SystemExit("unknown slug: " + ", ".join(unknown))
    slugs = args.slugs or [slug for slug in skills if slug != CHARACTER]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for slug in slugs:
        with_character = slug != CHARACTER and args.character_ref is not None
        refs = [args.style_ref] + ([args.character_ref] if with_character else [])
        prompt = build_prompt(skills[slug], with_character)
        for take in range(1, args.variants + 1):
            name = f"{slug}.jpg" if args.variants == 1 else f"{slug}-{take}.jpg"
            jobs.append((args.out_dir / name, prompt, refs))

    def run(job: tuple[pathlib.Path, str, list[pathlib.Path]]) -> str:
        out, prompt, refs = job
        try:
            out.write_bytes(generate(prompt, refs, key))
            return f"saved {out}"
        except RuntimeError as error:
            return f"FAILED {out.name}: {error}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(run, jobs):
            print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
