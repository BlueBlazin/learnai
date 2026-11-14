# BUG: reuses an entry only when all of its tokens are a prefix of the prompt; an entry that diverges from the prompt part-way is ignored instead of truncated.
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
    for key in store:
        shared = common_prefix_len(key, prompt)
        if shared == len(key) and shared > best:
            best_key, best = key, shared
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
