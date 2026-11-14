import jax.numpy as jnp

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
    # The stored entry that shares the most leading tokens with the prompt.
    best_key, best = None, 0
    for key in store:
        shared = common_prefix_len(key, prompt)
        if shared > best:
            best_key, best = key, shared
    # Keep at least one prompt token to feed: without a call there are no logits.
    reused = min(best, len(prompt) - 1)

    if reused > 0:
        # The entry may go on differently after the shared part: cut that off.
        cache = store[best_key].truncate(reused)
        logits, cache = lm(prompt[reused:], cache)
    else:
        logits, cache = lm(prompt)
    # `cache` now covers exactly the prompt. Caches are immutable, so the decode
    # steps below do not change what is stored.
    new_store = {**store, tuple(prompt): cache}

    generated = []
    while True:
        token = int(jnp.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            return generated, new_store
        logits, cache = lm([token], cache)
