Every parameter in this model lives in one of three kinds of tensors: a table with one
row per token, a matrix that maps `n_in` features to `n_out` features (plus, perhaps,
a bias of `n_out`), or a per-feature vector inside a norm. Before multiplying anything,
list the tensors of one block. What are `n_in` and `n_out` for each of the four
attention projections when there are fewer KV heads than query heads?

---

Work out two widths first: the query width `n_heads * head_dim` and the KV width
`n_kv_heads * head_dim`. Neither is `d_model` in general. $W_q$ goes from `d_model` to
the query width, $W_k$ and $W_v$ from `d_model` to the KV width, and $W_o$ from the
query width back to `d_model`. The MLP has three `d_model x d_ff` matrices when it is
gated (`"swiglu"`) and two otherwise. Compute one block, multiply by `n_layers`, and
only then add what exists once: the embedding, the final norm, and the head.

---

Per block, with `q = n_heads * head_dim` and `kv = n_kv_heads * head_dim`:

```python
attention = 2 * d * q + 2 * d * kv          # + q + 2 * kv + d   if attn_bias
mlp = 3 * d * d_ff                          # + 2 * d_ff + d     if mlp_bias   (swiglu)
mlp = 2 * d * d_ff                          # + d_ff + d         if mlp_bias   (gelu)
```

There are `2 * n_layers + 1` norms of `d` parameters each (`2 * d` with `norm_bias`).
The head counts `vocab_size * d` only when `tie_embeddings` is false: a tied head is
the embedding tensor under a second name, not a second tensor.
