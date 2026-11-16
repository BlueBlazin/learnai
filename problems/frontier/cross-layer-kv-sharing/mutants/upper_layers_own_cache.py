# BUG: the upper layers attend over keys and values of their own and cache them, which is the ordinary model: a different function with different logits, and twice the stored KV.
from learnai.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache)
    return lm.head(lm.norm(h))[-1], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, lm.n_layers, cache)
    return lm.head(lm.norm(h))[0], cache
