# BUG: makes len(prompt) // chunk_size calls and puts the leftover tokens into the last one, which is then larger than chunk_size.
import torch

from whetstone.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    n_calls = max(1, len(prompt) // chunk_size)
    cache = None
    for i in range(n_calls):
        end = len(prompt) if i == n_calls - 1 else (i + 1) * chunk_size
        logits, cache = lm(prompt[i * chunk_size : end], cache)
    return logits[-1], cache
