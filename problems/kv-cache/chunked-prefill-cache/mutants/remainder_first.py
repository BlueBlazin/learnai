# BUG: splits from the end of the prompt, so the short chunk comes first instead of last.
import torch

from whetstone.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    cache = None
    first = len(prompt) % chunk_size or chunk_size
    logits, cache = lm(prompt[:first], cache)
    for start in range(first, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    return logits[-1], cache
