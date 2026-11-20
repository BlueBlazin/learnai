Write down, for a call with `S` cached tokens and `T` new ones, the shape of the index
score matrix and its mask. The queries are the `T` new tokens. Which keys can new
token `i` be scored against, and where do those keys live at the moment you compute
the scores: in the cache you were given, in this call's `x_new`, or both?

---

Do the two appends first, then everything else on the grown caches:

```python
index_keys = concatenate((index_keys, dsa.index_key(layer, x_new)))  # (S + T, D)
q, k_new, v_new = dsa.qkv(layer, x_new)
store = store.append(k_new, v_new)  # S + T rows
```

Now the scores have shape `(T, S + T)`: `index_query` of the new rows against all of
`index_keys`. New row `i` is the token at position `S + i`, so it may select columns
`s <= S + i`: every cached token, the new tokens before it, and itself. That mask is
neither square nor "everything": `arange(S + T)[None, :] <= (S + arange(T))[:, None]`.

---

Selection as index lists: mask the scores with `-inf`, take a stable descending
argsort, keep the first `k` columns, and replace picks that are not visible by `-1`
(they occur only in rows with fewer than `k` visible tokens):

```python
order = argsort(masked, descending, stable)[:, :k]
idx = where(take_along_axis(visible, order, axis=1), order, -1)
```

Then the code of `gathered-sparse-attention` on the grown store:
`store.gather(idx, valid=idx >= 0)`, scores `q . k_rows * head_dim ** -0.5`, `-inf` in
the unused slots, softmax over slots, weighted sum of `v_rows`, `dsa.out`. Return
`(out, (index_keys, store))` with the grown `index_keys` and the appended store.
