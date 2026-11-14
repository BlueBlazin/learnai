Picture the cache as tensors with named axes rather than as a formula. For one layer,
which tensors are kept, and what are their axes? Which of those axes grow when another
request arrives, which when another token is generated, and which are fixed by the
architecture? In particular: after grouped-query attention, how many distinct key
vectors does a layer produce per token?

---

One layer keeps two tensors, keys and values, each of shape
`(batch, n_kv_heads, tokens, head_dim)`. So one token of one sequence costs
`2 * n_kv_heads * head_dim` elements in each layer that keeps it, and an element costs
1, 2, or 4 **bytes** depending on `dtype` (a small dict from name to size is enough).
`kv_bytes_per_token` multiplies that by `n_layers`. For `kv_bytes`, layers no longer
all hold the same number of tokens, so do not multiply by `n_layers`: add up, layer by
layer, how many tokens each one holds.

---

A layer holds `min(context, window_layers[layer])` tokens if it is a key of
`window_layers`, and `context` tokens otherwise:

```python
layer_tokens = sum(min(context, window_layers.get(layer, context))
                   for layer in range(cfg["n_layers"]))
```

The answer is `batch * layer_tokens * 2 * n_kv_heads * head_dim * itemsize`. Check it
on the Gemma example: `5 * 8192 + 29 * 1024 = 70656` layer-tokens, times `4096` bytes
per layer-token.
