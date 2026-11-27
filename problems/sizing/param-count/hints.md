Every parameter in this model lives in one of three tensor types: a lookup table with one row per token, a weight matrix mapping `n_in` features to `n_out` features (plus an optional bias of length `n_out`), or a per-feature vector inside a norm layer. Before multiplying anything, list all tensors in a single block and write down `n_in` and `n_out` for each of the four attention projections when `n_kv_heads < n_heads`.

---

Compute two widths first: the query width `n_heads * head_dim` and the KV width `n_kv_heads * head_dim`. Neither equals `d_model` in general. $W_q$ maps from `d_model` to the query width, $W_k$ and $W_v$ map from `d_model` to the KV width, and $W_o$ maps from the query width back to `d_model`. The MLP has three matrices between `d_model` and `d_ff` when `mlp_type == "swiglu"`, and two otherwise. Sum one block, multiply by `n_layers`, and then add the non-repeated components: the embedding table, the final norm, and the output head.

---

Per block, with `d = d_model`, `q = n_heads * head_dim`, and `kv = n_kv_heads * head_dim`:

```python
attention = 2 * d * q + 2 * d * kv          # + q + 2 * kv + d   if attn_bias
mlp = 3 * d * d_ff                          # + 2 * d_ff + d     if mlp_bias   (swiglu)
mlp = 2 * d * d_ff                          # + d_ff + d         if mlp_bias   (gelu)
```

There are `2 * n_layers + 1` norms of `d` parameters each (`2 * d` when `norm_bias` is `True`). The head counts `vocab_size * d` only when `tie_embeddings` is `False`, because a tied head reuses the embedding tensor instead of allocating a second one.
