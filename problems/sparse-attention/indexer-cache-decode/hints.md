For a call with `S` cached tokens and `T` new tokens, write down the shape of the index score matrix and its causal mask. The queries are the `T` new tokens. Which keys can new token `i` score against, and where do those keys live when you compute the scores?

---

Do both appends first, then run scoring and attention on the grown caches:

```python
index_keys = concatenate((index_keys, dsa.index_key(layer, x_new)))  # (S + T, D)
q, k_new, v_new = dsa.qkv(layer, x_new)
store = store.append(k_new, v_new)  # S + T rows
```

The index scores now have shape `(T, S + T)`, computed from `index_query` on `x_new` against all of `index_keys`. New row `i` is at position `S + i`, so it may select columns `s <= S + i` (every cached token, earlier new tokens in the chunk, and itself). Build that mask with `arange(S + T)[None, :] <= (S + arange(T))[:, None]`.

---

Mask invisible scores with `-inf`, run a stable descending `argsort`, keep the first `k` columns, and replace any invisible pick with `-1` (this happens only in rows with fewer than `k` visible tokens):

```python
order = argsort(masked, descending, stable)[:, :k]
idx = where(take_along_axis(visible, order, axis=1), order, -1)
```

Then run the `gathered-sparse-attention` step on the grown store: call `store.gather(idx, valid=idx >= 0)`, compute `q . k_rows * head_dim ** -0.5`, set unused slots to `-inf`, apply softmax over slots, take the weighted sum of `v_rows`, and apply `dsa.out`. Return `(out, (index_keys, store))` using the grown `index_keys` and the appended `store`.
