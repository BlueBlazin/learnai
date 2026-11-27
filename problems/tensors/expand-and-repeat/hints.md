For a tiny input, write down the sequence of source indices the output should read. For `repeat_kv_heads` with 2 groups and `n_rep = 2`, the output heads should read groups `0, 0, 1, 1`. Check which function in your framework turns `[0, 1]` into `0, 0, 1, 1` and which produces `0, 1, 0, 1` by testing both on `arange(2)` first.

---

- `repeat_kv_heads` and `expand_for_beams` perform the same operation on different axes, repeating each element in place along axis `1` and axis `0` respectively.
- In PyTorch, use `repeat_interleave` (`repeat` tiles instead). In JAX and MLX, use `repeat` (`tile` tiles instead). Always specify the axis, because omitting it flattens the tensor first.
- `tile_positions` adds a new leading axis of size `1` and expands along it. In PyTorch and MLX, expanding or broadcasting that axis avoids copying memory.

---

The key lines, in PyTorch:

```
k.repeat_interleave(n_rep, dim=1)       # repeat_kv_heads

x.repeat_interleave(K, dim=0)           # expand_for_beams

pos[None, :].expand(B, -1)              # tile_positions
```

In JAX: `jnp.repeat(k, n_rep, axis=1)`, `jnp.repeat(x, K, axis=0)`, and `jnp.broadcast_to(pos, (B, pos.shape[0]))`. MLX provides the same three functions under `mx`.
