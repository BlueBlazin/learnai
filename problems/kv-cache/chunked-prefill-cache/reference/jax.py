import jax

from whetstone.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[jax.Array, KVCache]:
    cache = None
    # Consecutive slices: every chunk is full except possibly the last one, and the
    # range never produces an empty slice.
    for start in range(0, len(prompt), chunk_size):
        # The cache carries everything before `start`, so only the chunk goes in.
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    # Only the final chunk's last row scores the token after the prompt.
    return logits[-1], cache
