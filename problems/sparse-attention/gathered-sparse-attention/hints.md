`idx` is rectangular, while each query's selection can have a different size, so unused slots are padded with `-1`. A padded slot must neither be read (which affects the read counter) nor participate in the softmax (which affects the output). Which argument to `store.gather` skips reading padded slots, what values does it return for those slots, and what score would those values get if left unmasked?

---

Compute `valid = idx >= 0` once and reuse it twice:

- Call `k_rows, v_rows = store.gather(idx, valid=valid)` to read only the used slots. Both returned tensors have shape `(T, k, H, d)`, with slot `j` of query `t` at `[t, j]`.
- Compute `(T, H, k)` attention scores with `einsum("thd,tjhd->thj", q, k_rows) * d ** -0.5`. An unread slot holds a zero key and would score `0`, which is not a small score when real scores are negative. Replace unused-slot scores with `-inf` before softmax, using `valid[:, None, :]` to broadcast across the head axis.

---

After applying softmax over the slot axis, compute the output with `einsum("thj,tjhd->thd", weights, v_rows)`.

One edge case remains when a query has no used slots. All of its scores are `-inf`, so softmax computes `0 / 0` and yields `NaN`. Mask the weights once more:

```python
weights = where(valid[:, None, :], weights, 0.0)
```

This replaces `NaN` entries on completely empty rows with `0.0` while leaving all other rows unchanged, because unused slots on non-empty rows already have weight zero.
