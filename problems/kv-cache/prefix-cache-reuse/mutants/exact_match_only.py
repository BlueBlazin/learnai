# BUG: reuses a cache only when the whole prompt is a key of the store; a prompt that extends or partly matches an entry is prefilled from scratch.
import torch

from whetstone.kit import KVCache, TinyLM


def common_prefix_len(a, b) -> int:
    """The number of leading tokens `a` and `b` share."""
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def generate_with_store(
    lm: TinyLM, store: dict[tuple[int, ...], KVCache], prompt: list[int], max_new_tokens: int
) -> tuple[list[int], dict[tuple[int, ...], KVCache]]:
    best_key, best = None, 0
    if tuple(prompt) in store:
        best_key, best = tuple(prompt), len(prompt)
    reused = min(best, len(prompt) - 1)

    if reused > 0:
        cache = store[best_key].truncate(reused)
        logits, cache = lm(prompt[reused:], cache)
    else:
        logits, cache = lm(prompt)
    new_store = {**store, tuple(prompt): cache}

    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            return generated, new_store
        logits, cache = lm([token], cache)
