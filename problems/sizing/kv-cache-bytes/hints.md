Picture the cache as tensors with named axes rather than a formula. For a single layer, which tensors are kept and what are their axes? Which axes grow when another request arrives, which grow when another token is generated, and which are fixed by the architecture? With grouped-query attention, how many distinct key vectors does a layer produce per token?

---

Each layer keeps two tensors (keys and values), both of shape `(batch, n_kv_heads, tokens, head_dim)`. One token of one sequence therefore takes `2 * n_kv_heads * head_dim` elements per layer, and each element takes 1, 2, or 4 **bytes** depending on `dtype` (a small lookup dict works well). `kv_bytes_per_token` multiplies that per-layer size by `n_layers`. For `kv_bytes`, layers can hold different numbers of tokens, so sum the token count layer by layer instead of multiplying by `n_layers`.

---

Layer `layer` stores `min(context, window_layers[layer])` tokens if `layer` is in `window_layers`, and `context` tokens otherwise:

```python
layer_tokens = sum(min(context, window_layers.get(layer, context))
                   for layer in range(cfg["n_layers"]))
```

Multiply by the batch size and bytes per layer-token: `batch * layer_tokens * 2 * n_kv_heads * head_dim * itemsize`. On the Gemma example at context 8192, this gives `5 * 8192 + 29 * 1024 = 70656` layer-tokens times `4096` bytes per layer-token.
