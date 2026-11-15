# BUG: the upper layers read the keys and values of layer 0 instead of layer share_from - 1, which is the same layer only when share_from is 1.
from whetstone.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    h, cache = lm.layers(h[-1:], share_from, lm.n_layers, cache, kv_from=0)
    return lm.head(lm.norm(h))[0], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache, kv_from=0)
    return lm.head(lm.norm(h))[0], cache
