Write out the mask for `T = 5` and `window = 3` by hand, row by row, where row `i` is a query and column `j` is a key. For row 4, list which columns are `True` and count them. There must be exactly three, including the diagonal. Express the first and last `True` column indices in terms of `i` and `window`.

---

The mask combines two conditions on each index pair `(i, j)`:

- **Causal:** `j <= i` (no future keys).
- **Recent:** `j > i - window` (no keys older than the window).

Build a column of query indices with `arange(T)[:, None]` and a row of key indices with `arange(T)[None, :]`. Comparing them broadcasts to `(T, T)`, and combining the two boolean grids with `&` requires no loops and no special cases for early rows or `window > T`. Whenever `i - window` is negative, every `j >= 0` automatically satisfies `j > i - window`.

Then run standard scaled dot-product attention with this mask instead of a triangular causal mask. Compute the scaled scores, fill positions where the mask is `False` with `-inf`, apply `softmax` over the last axis, and multiply by `v`.

---

```python
i = arange(T)[:, None]
j = arange(T)[None, :]
mask = (j <= i) & (j > i - window)
```

The strict `>` ensures each full row has exactly `window` visible keys (`i - window + 1, ..., i`). Using `>=` would include `window + 1` keys. As a sanity check, when `window = 1`, only `j = i` is `True`, so every token attends only to itself and the attention output equals `v`.

Apply the mask **before** `softmax` by setting hidden scores to `-inf`, rather than zeroing weights afterward. The `(T, T)` mask broadcasts automatically across the batch and head dimensions of the `(B, H, T, T)` score tensor.
