After the prompt is processed, ask which intermediate outputs are ever used again. A layer's output at position $i$ can matter in only two ways: as input to the next layer at position $i$, or through its cached keys and values that later positions attend to. Which of those two exists for a layer that stores no keys and values?

---

Split the layer stack at `share_from`. Run the lower layers as an ordinary cached model with `lm.layers(h, 0, share_from, cache)` on every new token. Run the upper layers as readers with `lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)`, passing only the rows whose logits you actually need. During prefill, that is just the last row `h[-1:]`, which aligns with the last cached position in layer `share_from - 1`.

---

```
prefill_shared:
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, s)                                  # s * P layer-tokens
    h, cache = lm.layers(h[-1:], s, L, cache, kv_from=s - 1)       # L - s
    return lm.head(lm.norm(h))[0], cache

decode_step:
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, s, cache)
    h, cache = lm.layers(h, s, L, cache, kv_from=s - 1)            # use the updated cache
    return lm.head(lm.norm(h))[0], cache
```

In `decode_step`, both the order and the cache variable matter. The lower layers must run first, and the upper layers must receive the new `cache` returned by the lower layers so the new position can attend to its own keys and values.
