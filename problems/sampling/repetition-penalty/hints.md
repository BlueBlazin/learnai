Split the problem in two. First, a boolean table `seen` of shape `(B, V)` that says,
for every row and every vocabulary entry, whether that token is in the row's history.
Second, an elementwise formula that uses it. The second part is easy. Which tensor
operation writes values into positions given by a tensor of indices, for every row at
once?

---

Scatter along the vocabulary axis. Start from an all-`False` `(B, V)` table and write
into the columns named by `prev_tokens`, row by row: `scatter` in PyTorch,
`.at[rows, prev_tokens].set(...)` in JAX with `rows = arange(B)[:, None]`,
`put_along_axis` in MLX. Writing to the same position several times is harmless here,
which is exactly why duplicates in the history cost nothing. For padding, think about
what value you write rather than which positions you skip.

---

Write `prev_tokens != pad_id` instead of `True`: padding entries then write `False`
into the pad column, and real entries write `True` into theirs. Then

```
penalized = where(logits > 0, logits / penalty, logits * penalty)
out       = where(seen, penalized, logits)
```

Compute `penalized` from the original logits for the whole table and select with
`seen`; nothing is applied twice, and no input is modified.
