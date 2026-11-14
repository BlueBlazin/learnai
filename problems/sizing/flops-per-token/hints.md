Separate the work into three piles before writing any arithmetic: products with a
weight matrix, which cost the same for every position whatever the context; attention
products, whose cost depends on how many keys a query sees; and the head, which runs
once. For each pile, ask: how many times does it happen in a prefill of `t` tokens,
and how many times in one decode step?

---

Write three helpers. The per-position cost of a block is the four projections plus
the MLP matrices, each `2 * n_in * n_out`; mind that the key and value projections
are `d_model x (n_kv_heads * head_dim)` while the query and output projections use
`n_heads * head_dim`. The attention cost of **one query** against `n` keys is
`n_heads * (2 * head_dim * n + 2 * n * head_dim)`: scores, then mixing, for every
query head. The head is `2 * d_model * vocab_size`. A prefill has `t` positions and
`t` queries that each see `t` keys; a decode step has one position and one query that
sees `context` keys.

---

With `P` the per-position cost of a block and `A(n) = 4 * n_heads * head_dim * n`:

```python
prefill = n_layers * (t * P + t * A(t)) + head
decode = n_layers * (P + A(context)) + head
```

Two checks that catch most mistakes: `decode_flops(cfg, 1) == prefill_flops(cfg, 1)`,
and for the tiny example `P = 272`, `A(5) = 120`, `head = 80`.
