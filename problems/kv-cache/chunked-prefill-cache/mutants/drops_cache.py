# BUG: does not pass the cache to the next call, so every chunk is run as if it were the start of the sequence and the returned cache covers only the last chunk.
import torch

from learnai.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    for start in range(0, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size])
    return logits[-1], cache
