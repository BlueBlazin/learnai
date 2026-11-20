"""Regenerate one or all scenes from scenes.json.

Usage:
  GEMINI_API_KEY=... python3 generate.py --out-dir RAW_DIR --ref STYLE_IMAGE [slug ...]

With no slugs, every scene is generated. Each result is written to
RAW_DIR/<slug>.jpg, ready for postprocess.py. Keep RAW_DIR outside the repo.

The key is read from the GEMINI_API_KEY environment variable only, sent in a
header, and never printed. Standard library only.
"""

import argparse
import base64
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
            "imageConfig": {"aspectRatio": "21:9", "imageSize": "1K"},
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
        raise SystemExit(f"HTTP {error.code}: {error.read().decode()[:1000]}")
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("inlineData"):
                return base64.b64decode(part["inlineData"]["data"])
    raise SystemExit("no image in response: " + json.dumps(payload)[:1000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*", help="scene slugs; all scenes when omitted")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ref", type=pathlib.Path, action="append", required=True,
                        help="style reference image; repeat for more than one")
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY in the environment")

    scenes = {scene["slug"]: scene for scene in json.loads((HERE / "scenes.json").read_text())}
    unknown = [slug for slug in args.slugs if slug not in scenes]
    if unknown:
        raise SystemExit("unknown slug: " + ", ".join(unknown))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for slug in args.slugs or list(scenes):
        out = args.out_dir / f"{slug}.jpg"
        out.write_bytes(generate(scenes[slug]["prompt"], args.ref, key))
        print("saved", out, file=sys.stderr)


if __name__ == "__main__":
    main()
