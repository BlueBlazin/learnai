Split the problem into two steps. First, build a boolean table `seen` of shape `(B, V)`
that records, for every row and vocabulary entry, whether that token appears in the
row's history. Second, apply the elementwise penalty formula using `seen`. Which tensor
operation writes values into positions given by a tensor of indices for every row at
once?

---

Scatter along the vocabulary axis. Start from an all-`False` `(B, V)` table and write
into the columns given by `prev_tokens` for each row (`scatter` in PyTorch,
`.at[rows, prev_tokens].set(...)` in JAX with `rows = arange(B)[:, None]`, or
`put_along_axis` in MLX). Writing to the same position more than once is harmless here,
so duplicate tokens in the history require no special handling. For padding, think about
what value to write rather than which positions to skip.

---

Write `prev_tokens != pad_id` instead of `True`. Padding entries then write `False` into
the `pad_id` column, while real tokens write `True` into theirs. Next, compute

```
penalized = where(logits > 0, logits / penalty, logits * penalty)
out       = where(seen, penalized, logits)
```

Computing `penalized` from the original `logits` and selecting with `seen` ensures no
token is penalized twice and no input tensor is modified.
