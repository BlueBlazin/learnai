"""Regenerate the fixture of real-model-perplexity.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/foundations/real-model-perplexity/fixtures/make.py

Needs the small house model on disk (``ws assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is ``transformers`` (float32, CPU) with
its own tokenizer; the windowing is the loop of the Hugging Face guide "Perplexity of
fixed-length models", written out below with explicit targets.

``granite-4.0-350m.json`` holds, for each passage: the text, its token ids, the window
length and stride, the perplexity, the number of windows, and the smallest
log-probability of any scored token (the third passage contains a made-up word that
the model gives a probability near 1e-16, which is what rejects an epsilon inside the
logarithm). The kit's numpy model is run on the same windows as a cross-check.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from whetstone import assets
from whetstone.kit import RealModel
from whetstone.kit.realmodel import SMALL
from whetstone.problem import find_root

HERE = Path(__file__).parent
MAX_LEN = 16
PASSAGES = [
    (
        (
            "A whetstone is a fine-grained stone used to sharpen the edges of steel tools such "
            "as knives, chisels, and plane blades. The stone is wetted with water or oil "
            "before use."
        ),
        8,
    ),
    (
        (
            "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n"
            "print([fib(i) for i in range(10)])"
        ),
        4,
    ),
    (
        (
            "The capital of France is Paris. The capital of Japan is zxqv Tokyo, and the "
            "capital of Italy is Rome, of course."
        ),
        12,
    ),
]


def windowed(logits_of, ids: list[int], max_len: int, stride: int) -> tuple[float, int, float]:
    """(perplexity, number of windows, smallest log-probability) in float64."""
    n = len(ids)
    logprobs: list[float] = []
    previous_end, windows = 0, 0
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        logp = torch.log_softmax(logits_of(ids[begin:end]).double(), dim=-1)
        windows += 1
        for position in range(max(previous_end, 1), end):
            logprobs.append(float(logp[position - 1 - begin, ids[position]]))
        previous_end = end
        if end == n:
            break
    assert len(logprobs) == n - 1
    return math.exp(-sum(logprobs) / (n - 1)), windows, min(logprobs)


def main() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    kit = RealModel(directory).lm()

    def reference(window: list[int]) -> torch.Tensor:
        with torch.no_grad():
            return model(torch.tensor([window])).logits[0]

    def kit_model(window: list[int]) -> torch.Tensor:
        return torch.from_numpy(np.asarray(kit(window)[0]))

    passages = []
    for text, stride in PASSAGES:
        ids = tokenizer(text).input_ids
        value, windows, lowest = windowed(reference, ids, MAX_LEN, stride)
        check, _, _ = windowed(kit_model, ids, MAX_LEN, stride)
        assert abs(check - value) < 1e-4 * value, (value, check)
        whole, _, _ = windowed(reference, ids, len(ids), len(ids) - 1)
        passages.append(
            {
                "text": text,
                "ids": ids,
                "max_len": MAX_LEN,
                "stride": stride,
                "perplexity": value,
                "windows": windows,
                "lowest_logprob": lowest,
            }
        )
        print(f"{len(ids)} tokens, stride {stride}: perplexity {value:.4f} in {windows} windows "
              f"(kit {check:.4f}; one unlimited window {whole:.4f}); lowest log-prob {lowest:.2f}")  # fmt: skip
    out = {"revision": SMALL.revision, "passages": passages}
    target = HERE / "granite-4.0-350m.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target.name} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
