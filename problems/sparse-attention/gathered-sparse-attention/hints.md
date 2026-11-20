`idx` is a rectangular tensor, but the selections are ragged: the unused slots are
padding. Two things must happen to a padded slot. It must not be read (the counter),
and it must not take part in the softmax (the result). Which argument of
`store.gather` does the first, and what does that slot's row look like afterward? What
score would that row get if you did nothing more?

---

Build `valid = idx >= 0` once and use it twice.

- `k_rows, v_rows = store.gather(idx, valid=valid)` reads exactly the used slots. Both
  results have shape `(T, k, H, d)`: slot `j` of query `t` is at `[t, j]`.
- The scores have shape `(T, H, k)`: `einsum("thd,tjhd->thj", q, k_rows)` times
  `d ** -0.5`. A slot that was not read holds a zero key and would score 0, which is
  not a small score. Replace the scores of unused slots with `-inf` before the
  softmax. `valid[:, None, :]` broadcasts over the head axis.

---

After the softmax over the slot axis, the output is
`einsum("thj,tjhd->thd", weights, v_rows)`.

One case is left: a query with no used slot. All of its scores are `-inf`, and a
softmax over them is `0 / 0`, NaN. Select the weights once more,

```python
weights = where(valid[:, None, :], weights, 0.0)
```

which replaces the NaN of such a row by zeros and changes nothing else, because the
weights of unused slots are already zero in every other row.
