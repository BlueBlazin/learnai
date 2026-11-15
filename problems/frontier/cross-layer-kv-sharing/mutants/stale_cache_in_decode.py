# BUG: decode_step hands the upper layers the cache from before the step, so they read layer share_from - 1 without the new token's keys and values and the new position cannot attend to itself.
from whetstone.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    h, cache = lm.layers(h[-1:], share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    h = lm.embed([token], offset=cache.length)
    h, new_cache = lm.layers(h, 0, share_from, cache)
    h, _ = lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], new_cache
