# BUG: prefill feeds the prompt to the lower layers one token at a time instead of in one call: the same layer-tokens and logits, but share_from layer calls per prompt token.
from whetstone.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    cache = lm.empty_cache()
    for position, token in enumerate(prompt):
        h, cache = lm.layers(lm.embed([token], offset=position), 0, share_from, cache)
    h, cache = lm.layers(h[-1:], share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache
