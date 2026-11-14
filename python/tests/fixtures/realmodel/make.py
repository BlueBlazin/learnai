"""Regenerate the reference fixtures for the real-model kit tests.

    uv run --project python --extra models python python/tests/fixtures/realmodel/make.py

Needs the small house model on disk (``ws assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU); the
token ids come from ``tokenizers``. What is stored is small on purpose: for every
prompt the ids, the argmax at every position, the logits of every position at 64 fixed
vocabulary columns, the top 32 logits of the last position, and eight columns of the
hidden state after the embedding, block 1, and block 14.

This is also the pattern for a problem's own ``fixtures/make.py``.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from whetstone import assets
from whetstone.kit.realmodel import SMALL
from whetstone.problem import find_root

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:",
    "Whetstone sharpens 🗡️ blades; 東京 is in Japan.",
]
#: The vocabulary columns whose logits are stored for every position.
COLUMNS = np.random.default_rng(350).choice(100352, size=64, replace=False)
HIDDEN_LAYERS = (0, 1, 14)


def main() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    out: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(SMALL.revision)}
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            result = model(ids, output_hidden_states=True)
        logits = result.logits[0].numpy()
        top = np.argsort(-logits[-1])[:32]
        out[f"ids_{index}"] = ids[0].numpy().astype(np.int32)
        out[f"argmax_{index}"] = logits.argmax(-1).astype(np.int32)
        out[f"logits_{index}"] = logits[:, COLUMNS]
        out[f"top_ids_{index}"] = top.astype(np.int32)
        out[f"top_logits_{index}"] = logits[-1, top]
        for layer in HIDDEN_LAYERS:
            out[f"hidden_{index}_{layer}"] = result.hidden_states[layer][0, :, :8].numpy()
    target = Path(__file__).with_name(f"{SMALL.name}.npz")
    np.savez_compressed(target, **out)
    (Path(__file__).with_name("prompts.json")).write_text(
        json.dumps(PROMPTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
