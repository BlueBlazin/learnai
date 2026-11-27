Split the work into three groups before writing any formulas. First, weight-matrix products cost the same for every position regardless of context length. Second, attention products depend on how many keys each query attends to. Third, the output head runs only once. For each group, count how many times it executes during a prefill of `t` tokens and during one decode step.

---

Write three helper functions. The per-position cost of one block is the four attention projections plus the MLP matrices, each costing `2 * n_in * n_out`. Keep in mind that the key and value projections have shape `d_model x (n_kv_heads * head_dim)`, while the query and output projections use `n_heads * head_dim`. For attention, **one query** attending over `n` keys costs `n_heads * (2 * head_dim * n + 2 * n * head_dim)` for scores and value mixing across all query heads. The output head costs `2 * d_model * vocab_size`. A prefill has `t` positions and `t` queries that each attend over `t` keys, while a decode step has one position and one query that attends over `context` keys.

---

Let `P` be the per-position cost of one block and `A(n) = 4 * n_heads * head_dim * n`.

```python
prefill = n_layers * (t * P + t * A(t)) + head
decode = n_layers * (P + A(context)) + head
```

Two sanity checks catch most bugs. First, `decode_flops(cfg, 1)` must equal `prefill_flops(cfg, 1)`. Second, for the `tiny` example, `P = 272`, `A(5) = 120`, and `head = 80`.
