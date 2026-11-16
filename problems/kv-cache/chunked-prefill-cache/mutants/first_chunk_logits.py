# BUG: keeps the logits of the first chunk instead of the last one, so the returned row scores the token after the first chunk, not after the prompt.
import torch

from learnai.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    cache = None
    last_logits = None
    for start in range(0, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
        if last_logits is None:
            last_logits = logits[-1]
    return last_logits, cache
