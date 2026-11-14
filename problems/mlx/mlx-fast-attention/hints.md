Compare `n_cached + T` with `S`. When they are equal, every slot holds a token: which
of the kernel's built-in mask forms describes "token $i$ sees everything up to itself"
in that layout, and which describes "every token sees everything"? When they are not
equal, what is the set of slots a new token may read, written as a comparison between
a slot index and a number that depends on $i$?

---

A full buffer needs no mask array: `mask="causal"` when `causal` is true and there is
more than one new token, `mask=None` otherwise. With unused capacity, build a boolean
array that is `True` where slot $j$ may be read by new token $i$: `j <= n_cached + i`
for the causal case, `j < n_cached + T` otherwise. A boolean mask means "may attend"
and works for float16 and float32 inputs alike; an additive mask would have to be
created in the dtype of `q`.

Pass `k` and `v` unchanged even when they have fewer heads than `q`.

---

```python
steps, slots = q.shape[2], k.shape[2]
last = n_cached + mx.arange(steps)[:, None] if causal else n_cached + steps - 1
mask = mx.arange(slots)[None, :] <= last  # (T, S) or (1, S), True = may attend
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
```

That mask is correct for every layout. Use it when `n_cached + steps < slots`, and
the string or `None` when the buffer is full.
