from whetstone.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    # The lower layers see every prompt token: they are the only layers that store
    # keys and values, and the upper layers will read those of layer share_from - 1.
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    # The upper layers store nothing, so a prompt position they do not run on leaves
    # no gap behind. Only the last position is needed: its output scores the next token.
    h, cache = lm.layers(h[-1:], share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    # The new token sits at position cache.length (learned positions need the offset).
    h = lm.embed([token], offset=cache.length)
    # Lower layers first: layer share_from - 1 must have cached this position before
    # the upper layers read it, and they must read the *new* cache.
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache
