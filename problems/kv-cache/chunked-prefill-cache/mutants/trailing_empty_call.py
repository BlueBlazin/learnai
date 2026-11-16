# BUG: the loop bound is len(prompt) + 1, so a prompt whose length is a multiple of chunk_size ends with a call on an empty chunk.
import torch

from learnai.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    cache = None
    for start in range(0, len(prompt) + 1, chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    return logits[-1], cache
