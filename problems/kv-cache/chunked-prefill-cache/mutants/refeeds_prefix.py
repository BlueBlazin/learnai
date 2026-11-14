# BUG: re-feeds the whole prefix with every chunk instead of continuing from the cache; the logits are right but tokens_processed grows quadratically with the number of chunks.
import torch

from whetstone.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    for end in range(chunk_size, len(prompt) + chunk_size, chunk_size):
        logits, cache = lm(prompt[:end])
    return logits[-1], cache
