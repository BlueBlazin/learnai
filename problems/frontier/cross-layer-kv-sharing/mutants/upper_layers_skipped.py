# BUG: prefill treats the whole prompt as cache-only work and never runs the upper layers on the last position; the logits it returns come from the state after layer share_from - 1.
from learnai.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    return lm.head(lm.norm(h[-1:]))[0], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache
