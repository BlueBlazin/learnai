import torch

from whetstone.kit import LayerCache, LayeredLM


def prefill_shared(lm: LayeredLM, prompt: list[int], share_from: int) -> tuple:
    """Process a prompt with a model whose upper layers share the KV of one lower layer.

    lm: a LayeredLM with L = lm.n_layers layers.
        h = lm.embed(tokens, offset=n)        hidden states of tokens at positions n, n + 1, ...
        h, cache = lm.layers(h, a, b, cache)  layers a .. b - 1; each stores its own keys and values
        h, cache = lm.layers(h, a, b, cache, kv_from=j)
                                              layers a .. b - 1 reading layer j's keys and values
                                              and storing none; the rows of h are the last
                                              positions layer j has cached
        logits = lm.head(lm.norm(h))          (T, vocab_size)
    prompt: the prompt token ids, P >= 1.
    share_from: s with 1 <= s < L. Layers s .. L - 1 read the keys and values of layer s - 1.
    Returns (logits, cache): the next-token logits after the prompt, shape (vocab_size,),
    and a cache in which layers 0 .. s - 1 hold P tokens and layers s .. L - 1 hold none.
    """
    raise NotImplementedError


def decode_step(lm: LayeredLM, token: int, cache: LayerCache, share_from: int) -> tuple:
    """Feed one more token to the sequence that `cache` covers.

    token: the token id at position cache.length.
    cache: what prefill_shared or an earlier decode_step returned (never modified).
    Returns (logits, cache): the next-token logits after `token`, shape (vocab_size,),
    and the cache extended by one token in layers 0 .. s - 1.
    """
    raise NotImplementedError
