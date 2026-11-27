## The idea

A tensor operation picks its result type from the types of its operands before it
looks at any value. If the true result does not fit that type, the bits that do not
fit are dropped. So for every integer operation there are two questions: what is the
range of the true result, and what type will hold it.

**`dequantize`.** `q` is int8, and a Python int takes the type of the tensor it meets,
so `q - zero_point` is an int8 subtraction. Its true range is $[-255, 255]$. Whenever
the difference leaves $[-128, 127]$ it wraps by 256: `100 - (-100)` is `200 - 256 =
-56`. The fix is to change the type before subtracting. float32 represents every
integer up to $2^{24}$ exactly, so casting `q` to float32 first loses nothing, and
the result has to be float32 anyway.

**`saturating_add_u8`.** uint8 plus uint8 is uint8, and the true range is
$[0, 510]$. `200 + 100` is `300 - 256 = 44`. Clamping the uint8 sum at 255 cannot
help: every uint8 value is already at most 255. Add in int16, clamp there, and cast
back. The cast back is exact because the clamp has put every value in range.

**`block_and_offset`.** Floor and truncation differ exactly when the true quotient is
negative and not a whole number: $-7 / 3 = -2.33$ is $-3$ rounded down and $-2$
rounded toward zero. The remainder follows the quotient, since
`pos = block * size + offset` must hold: floor gives an offset in $[0, \text{size})$,
truncation gives $-1$ for position $-7$. A block table indexed by `block` and a
buffer indexed by `offset` need the floor pair.

## Reference solution (PyTorch)

```python
def dequantize(q, scale, zero_point):
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a, b):
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos, size):
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
```

In JAX and MLX the casts are `x.astype(...)`, the clamp is `minimum(total, 255)`, and
the division is `floor_divide` and `remainder`.

Other correct forms. For the sum, staying in uint8 and never overflowing:
`a + torch.minimum(b, 255 - a)`, since `255 - a` is the room left; or detecting the
wrap, `torch.where(a + b < a, 255, a + b)`, since a wrapped sum is smaller than
either operand. For the division, `pos // size` and `pos % size` are the floor pair
in all three frameworks at the versions used here, and `jnp.divmod` and `mx.divmod`
return both.

## Common mistakes

- **Subtracting in int8.** `(q - zero_point) * scale` is the formula as written and
  wraps for half of the codes when `zero_point = -128`. It is right when
  `zero_point = 0`, which is the symmetric quantization most tests start with.
- **The zero point after the scale.** `q * scale - zero_point` subtracts a code from
  a real number. The two agree when `scale = 1` or `zero_point = 0`.
- **float64 for safety.** `q.to(torch.float64)` gives the right numbers in the wrong
  type, and a float64 tensor makes every float32 tensor it meets float64 in PyTorch.
  JAX truncates a request for float64 to float32 with a warning, and MLX raises for
  float64 on the GPU.
- **Adding in uint8.** `a + b`, with or without a `clamp` after it.
- **Clamping after the wrap.** `torch.clamp(a + b, max=255)` returns `a + b`.
- **Forgetting to cast back.** The clamped int16 tensor has the right values and
  twice the memory, and it is not what a uint8 buffer or a later uint8 operation
  expects. The result type is part of this function's contract.
- **Truncating division.** `torch.div(pos, size, rounding_mode="trunc")`,
  `int(pos / size)`, and a float quotient cast to an integer all round toward zero.
  Block `0` then covers positions `-(size - 1)` to `size - 1`, almost twice as many
  as any other block.
- **`fmod` for the offset.** Its result has the sign of `pos`, so negative positions
  get negative offsets, which index from the end of a buffer without an error.
- **Dividing in float.** `torch.floor(pos / size)` rounds down correctly and divides
  inexactly: float32 holds `999999999` as `1000000000`, so position `999999999` with
  `size = 1000` lands in block `1000000` where `999999` is right.

## Complexity

One pass over the input for each function, $O(n)$. The int16 intermediate of the sum
is twice the size of its inputs while it lives.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| int8 tensor − Python int in $[-128, 127]$ | int8, wraps | int8, wraps | int8, wraps |
| int8 tensor − Python int 200 | int8, wraps | int8, wraps | `ValueError` |
| int8 tensor − int32 tensor with no axes | int8, wraps | int32 | int32 |
| int8 tensor − int32 tensor with one axis | int32 | int32 | int32 |
| uint8 + uint8 | uint8, wraps | uint8, wraps | uint8, wraps |
| uint8 + int8 | int16 | int16 | int16 |
| uint8 + uint16 | `RuntimeError` | uint16 | uint16 |
| integer tensor × Python float | float32 | float32 | float32 |
| `sum` of a uint8 tensor | int64 | uint32 | uint32 |
| `cumsum` of a uint8 tensor | int64 | uint8, wraps | uint8, wraps |
| int16 `300` cast to uint8 | 44 | 44 | 44 |
| float32 `300.0` cast to uint8 | 44 | 255 | 255 |
| float32 `-1.0` cast to uint8 | 255 | 0 | 0 |
| float32 `-2.7` cast to int32 | -2 | -2 | -2 |
| `-7 // 3` and `-7 % 3` on integer tensors | -3 and 2 | -3 and 2 | -3 and 2 |
| floor division by name | `torch.div(a, b, rounding_mode="floor")`, `torch.floor_divide` | `jnp.floor_divide` | `mx.floor_divide` |
| truncating division | `torch.div(a, b, rounding_mode="trunc")` | `jax.lax.div` | none found; `mx.trunc(a / b)` goes through float |
| remainder, sign of the divisor | `torch.remainder` | `jnp.remainder` | `mx.remainder` |
| remainder, sign of the dividend | `torch.fmod` | `jnp.fmod` | no `mx.fmod` |
| quotient and remainder together | no `torch.divmod` | `jnp.divmod` | `mx.divmod` |
| integer division by zero | `RuntimeError` | no error, arbitrary values | no error, arbitrary values |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The promotion rules agree on everything this problem needs: a Python int never
widens a tensor, and equal types stay as they are. Three differences are worth
knowing. In PyTorch a tensor with no axes behaves like a Python number in promotion,
so an int32 zero point stored as a 0-d tensor still leaves `q - zero_point` in int8,
where JAX and MLX widen to int32. MLX refuses a Python int that does not fit the
tensor's type, which turns one silent wrap into an error. And an out-of-range float
cast to an integer type wraps in PyTorch and saturates in JAX and MLX on this
machine, so clamp while still in float, as Affine int8 Quantization does.
