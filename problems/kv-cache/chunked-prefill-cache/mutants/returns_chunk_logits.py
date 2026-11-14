# BUG: returns the logits of the whole last chunk, shape (T, vocab_size), instead of its last row.
import torch

from whetstone.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    cache = None
    for start in range(0, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    return logits, cache
