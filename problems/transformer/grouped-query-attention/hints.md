Set attention aside for a moment and focus on the head axis. With $H = 6$ query heads
and $H_{kv} = 2$ key/value heads, write down which key/value head each query head
`0, 1, 2, 3, 4, 5` must read. If you expand `k` to 6 heads so that head `h` pairs with
query head `h`, in what order do the two original heads appear along that axis?

---

The expanded head order is `0, 0, 0, 1, 1, 1`, with each key/value head repeated
$G = H / H_{kv}$ times **in place**. Once `k` and `v` have shape `(B, H, S, d)` in that
order, the rest is standard scaled dot-product attention:

1. Compute `scores = q @ k^T / sqrt(d)`, where `k^T` swaps the last two axes, giving
   shape `(B, H, T, S)`.
2. Set `scores` to `-inf` where `mask` is `False`. The boolean `mask` of shape `(T, S)`
   broadcasts over batch and heads, with `True` meaning "may attend".
3. Apply softmax over the last axis, then multiply by `v`.

Every framework has two operations that extend an axis, and they produce different
orders. One repeats each element in place (`0, 0, 0, 1, 1, 1`), while the other repeats
the whole axis (`0, 1, 0, 1, 0, 1`). You need the first.

---

Use `repeat_interleave(G, dim=1)` in PyTorch, or `repeat(k, G, axis=1)` in JAX and MLX,
to repeat each head in place. (`tile` and PyTorch's `Tensor.repeat` repeat the entire
axis instead.)

```python
G = q.shape[1] // k.shape[1]
k = k.repeat_interleave(G, dim=1)  # (B, H_kv, S, d) -> (B, H, S, d)
v = v.repeat_interleave(G, dim=1)
```

You can get the exact same result without copying by reshaping `q` to
`(B, H_kv, G, T, d)`, inserting a length-1 axis into `k` and `v` after the head axis,
running attention with broadcasting, and reshaping the output back to `(B, H, T, d)`.
Both approaches handle $H_{kv} = 1$ ($G = H$) and $H_{kv} = H$ ($G = 1$) without any
special case.
