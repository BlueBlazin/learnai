# BUG: ignores chunk_size and feeds the prompt one token per call; the result is right but it makes len(prompt) forward calls.
import torch

from learnai.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    cache = None
    for token in prompt:
        logits, cache = lm([token], cache)
    return logits[-1], cache
