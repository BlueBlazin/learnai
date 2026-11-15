After the prompt has been processed, which results of the forward pass does anyone
use later? A layer's output at position $i$ can matter in two ways: as the input of
the next layer at position $i$, and, through its keys and values, as something a
later position attends to. Which of the two exists for a layer that stores no keys
and values?

---

Split the stack at `share_from`. The lower part is an ordinary cached model: run
`lm.layers(h, 0, share_from, cache)` on every new token. The upper part is
`lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)`, and it only
needs the rows whose output you will read. In the prefill that is the last row,
`h[-1:]`; the reader rows are taken to be the last positions the source layer has
cached, which is exactly what the last row is.

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
    h, cache = lm.layers(h, s, L, cache, kv_from=s - 1)            # the cache from the line above
    return lm.head(lm.norm(h))[0], cache
```

The order in `decode_step` matters twice: the lower layers must run first, and the
upper layers must be given the cache the lower layers just returned. With the old
cache the new position cannot see its own keys and values.
