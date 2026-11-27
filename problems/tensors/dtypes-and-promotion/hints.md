For each function, find the minimum and maximum values of the intermediate result and compare them to the range of the dtype used to compute it. `q - zero_point` reaches $\pm 255$, while `int8` only holds $-128$ to $127$. `a + b` reaches `510`, while `uint8` only holds `0` to `255`. Check which dtype each operation actually runs in.

---

- `dequantize`: Because `q` is `int8` and `zero_point` is a Python int, `q - zero_point` runs in `int8`. Cast `q` to a type that holds $\pm 255$ before subtracting (`float32` works directly and matches the required output dtype).
- `saturating_add_u8`: Adding two `uint8` tensors wraps before any `clamp` or `minimum` sees the sum. Cast to a wider integer type first, clamp at `255`, and cast back to `uint8`.
- `block_and_offset`: Use floor division and a non-negative remainder for positive `size`. Test your framework's operators on `-7` and `3`. Floor gives `-3` and `2`, while truncation gives `-2` and `-1`. Keep the division in integers because `float32` loses precision on large positions.

---

The key lines in PyTorch:

```
(q.to(torch.float32) - zero_point) * scale                 # dequantize

total = a.to(torch.int16) + b.to(torch.int16)
torch.clamp(total, max=255).to(torch.uint8)                # saturating_add_u8

block = torch.div(pos, size, rounding_mode="floor")
offset = torch.remainder(pos, size)                        # block_and_offset
```

In JAX and MLX, use `q.astype(jnp.float32)`, `jnp.minimum(total, 255).astype(jnp.uint8)`, `jnp.floor_divide(pos, size)`, and `jnp.remainder(pos, size)` (with `mx` in place of `jnp` for MLX).
