Write down, for a tiny input, the list of source indices the result should read:
for `repeat_kv_heads` with 2 groups and `n_rep = 2`, the heads read groups
`0, 0, 1, 1`. Then check which of your framework's functions produces that list from
`[0, 1]`, and which produces `0, 1, 0, 1`. Try both on `arange(2)` before using
them on the real tensor.

---

- `repeat_kv_heads` and `expand_for_beams` are the same operation on different axes:
  each element repeated in place, along axis 1 and along axis 0.
- In PyTorch that is `repeat_interleave`, and `repeat` is the other one. In JAX and
  MLX it is `repeat`, and `tile` is the other one. All of them need the axis named:
  without it they flatten the tensor first.
- `tile_positions` adds an axis in front and repeats along it. Since the new axis has
  size 1, it can be expanded (broadcast), which in PyTorch and MLX copies nothing.

---

The key lines, in PyTorch:

```
k.repeat_interleave(n_rep, dim=1)       # repeat_kv_heads

x.repeat_interleave(K, dim=0)           # expand_for_beams

pos[None, :].expand(B, -1)              # tile_positions
```

In JAX: `jnp.repeat(k, n_rep, axis=1)`, `jnp.repeat(x, K, axis=0)`, and
`jnp.broadcast_to(pos, (B, pos.shape[0]))`. MLX has the same three functions under
`mx`.
