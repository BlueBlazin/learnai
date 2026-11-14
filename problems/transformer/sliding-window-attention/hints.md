Write out the mask for `T = 5` and `window = 3` by hand, row by row. Row `i` is a
query; a column `j` is a key. For row 4, which columns are `True`? Count them: there
must be exactly three, and the diagonal is one of them. Now look at the first and the
last `True` of that row. What are their column indices in terms of `i` and `window`?

---

The mask is the conjunction of two conditions on the index pair `(i, j)`:

- causal: `j <= i` (nothing from the future);
- recent: `j > i - window` (nothing older than the window).

Build a column of query indices, `arange(T)[:, None]`, and a row of key indices,
`arange(T)[None, :]`. Comparing them broadcasts to `(T, T)`, and the two boolean
matrices combine with `&`. No loop, and no special case for the first rows or for
`window > T`: when `i - window` is negative, every `j >= 0` already satisfies the
second condition.

The attention function is then the one you know, with this mask instead of the
triangular one: compute the scores, set them to `-inf` where the mask is `False`,
take the softmax over the last axis, multiply by `v`.

---

```python
i = arange(T)[:, None]
j = arange(T)[None, :]
mask = (j <= i) & (j > i - window)
```

The strict `>` is what makes the count come out as `window`: the visible keys are
`i - window + 1, ..., i`. With `>=` each row would have `window + 1` of them. Check
with `window = 1`: only `j = i` survives, every token attends to itself alone, and the
attention output must equal `v` exactly.

Mask the scores **before** the softmax (`-inf` where `mask` is `False`), not the
weights after it. The mask has shape `(T, T)` and broadcasts over the batch and head
axes of the `(B, H, T, T)` scores.
