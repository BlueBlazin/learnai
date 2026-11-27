For each function, write down the largest and smallest value the intermediate result
can take, and the range of the type it is computed in. `q - zero_point` reaches
$\pm 255$; int8 holds $-128$ to $127$. `a + b` reaches 510; uint8 holds 0 to 255.
What type is the intermediate result actually computed in, given the operands?

---

- `dequantize`: `q` is int8 and `zero_point` is a Python int, so `q - zero_point` is
  int8. Change the type of `q` before the subtraction, to one that holds $\pm 255$.
  The result must be float32 in the end anyway.
- `saturating_add_u8`: uint8 plus uint8 is uint8, and the sum has wrapped before any
  `clamp` or `minimum` sees it. Add in a wider type, limit the result to 255 there,
  and cast back to uint8.
- `block_and_offset`: you need the division that rounds down and the remainder that
  is never negative for a positive `size`. Check what your framework's operators do
  on `-7` and `3` before trusting them: floor gives `-3` and `2`, truncation gives
  `-2` and `-1`. Stay in integers; a float32 quotient is not exact for large
  positions.

---

The key lines, in PyTorch:

```
(q.to(torch.float32) - zero_point) * scale                 # dequantize

total = a.to(torch.int16) + b.to(torch.int16)
torch.clamp(total, max=255).to(torch.uint8)                # saturating_add_u8

block = torch.div(pos, size, rounding_mode="floor")
offset = torch.remainder(pos, size)                        # block_and_offset
```

In JAX and MLX: `q.astype(jnp.float32)`, `jnp.minimum(total, 255).astype(jnp.uint8)`,
and `jnp.floor_divide(pos, size)`, `jnp.remainder(pos, size)`, with the same names
under `mx`.
