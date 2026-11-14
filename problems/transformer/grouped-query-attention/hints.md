Forget attention for a moment and look only at the head axis. With $H = 6$ query heads
and $H_{kv} = 2$ key/value heads, write down, for query heads `0, 1, 2, 3, 4, 5`, the
index of the key/value head each one must read. If you turn `k` into a tensor with 6
heads so that head `h` of the new tensor is the right one for query head `h`, in what
order do the two original heads appear along that axis?

---

The order is `0, 0, 0, 1, 1, 1`: each key/value head repeated $G = H / H_{kv}$ times
**in place**. Once `k` and `v` have shape `(B, H, S, d)` in that order, the rest is the
attention you already know:

1. `scores = q @ k^T / sqrt(d)`, where `k^T` swaps the last two axes: `(B, H, T, S)`.
2. Set the scores to `-inf` where `mask` is `False`. The mask has shape `(T, S)` and
   broadcasts over batch and heads. It is boolean: `True` means "may attend".
3. Softmax over the last axis, then multiply by `v`.

Every framework has two ways to make a tensor longer along an axis, and they give
different orders. One repeats each element (`0, 0, 0, 1, 1, 1`); the other repeats the
whole axis (`0, 1, 0, 1, 0, 1`). You need the first.

---

The operation that repeats each head in place is `repeat_interleave(G, dim=1)` in
PyTorch, and `repeat(k, G, axis=1)` in JAX and MLX. (`tile`, and PyTorch's
`Tensor.repeat`, repeat the whole axis instead.)

```python
G = q.shape[1] // k.shape[1]
k = k.repeat_interleave(G, dim=1)  # (B, H_kv, S, d) -> (B, H, S, d)
v = v.repeat_interleave(G, dim=1)
```

An equivalent way that copies nothing: reshape `q` to `(B, H_kv, G, T, d)`, insert an
axis of length 1 into `k` and `v` after the head axis, attend with broadcasting, and
reshape the result back to `(B, H, T, d)`. Both give `G = H` for a single key/value
head and `G = 1` for standard attention without any special case.
