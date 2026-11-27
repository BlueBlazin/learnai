## The idea

A tensor operation chooses its output dtype from its operand dtypes before reading any values. If the exact result falls outside that dtype's range, the high bits are dropped silently. Before writing any integer expression, check the true range of the intermediate result and make sure the computation runs in a type wide enough to hold it.

**`dequantize`.** Because `q` is `int8` and `zero_point` is a Python int, `q - zero_point` runs in `int8`. The true difference lies in $[-255, 255]$, so any value outside $[-128, 127]$ wraps by 256. For example, `100 - (-100)` becomes `200 - 256 = -56`. Cast `q` before subtracting. Since `float32` represents every integer up to $2^{24}$ exactly and the output must be `float32` anyway, casting `q` to `float32` first is both exact and direct.

**`saturating_add_u8`.** Adding two `uint8` tensors produces a `uint8` result, even though the exact sum ranges over $[0, 510]$. Thus `200 + 100` wraps to `300 - 256 = 44`. Clamping after adding in `uint8` does nothing because every `uint8` value is already at most 255. Instead, cast both operands to `int16`, add, clamp at 255, and cast back to `uint8`. The final cast is lossless because the clamp keeps every value in $[0, 255]$.

**`block_and_offset`.** Floor division and truncating division disagree whenever the true quotient is negative and non-integer. For `-7 / 3 = -2.33...`, floor gives `-3` while truncation gives `-2`. Because `pos = block * size + offset` must hold, floor division yields an offset in $[0, \text{size})$, whereas truncation yields `-1` for position `-7`. Indexing a block table by `block` and a buffer by `offset` requires the floor pair.

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

In JAX and MLX, replace `.to(...)` with `.astype(...)`, use `minimum(total, 255)` to clamp, and use `floor_divide` and `remainder`.

Other valid implementations exist. You can stay in `uint8` without overflowing by writing `a + torch.minimum(b, 255 - a)`, where `255 - a` is the remaining headroom. You can also detect the wrap with `torch.where(a + b < a, 255, a + b)`, because a wrapped unsigned sum is always smaller than either operand. For `block_and_offset`, `pos // size` and `pos % size` compute the floor pair in all three frameworks at the versions tested here, and `jnp.divmod` and `mx.divmod` return both at once.

## Common mistakes

- **Subtracting in `int8`.** Writing `(q - zero_point) * scale` directly wraps half of the codes when `zero_point = -128`. It happens to pass when `zero_point = 0`, which is the symmetric quantization case most tests try first.
- **Applying `zero_point` after scaling.** `q * scale - zero_point` subtracts a quantized code from a scaled float. It only agrees with the true formula when `scale = 1` or `zero_point = 0`.
- **Casting to `float64`.** `q.to(torch.float64)` produces the right numbers in the wrong dtype, and in PyTorch a `float64` tensor promotes every `float32` tensor it touches to `float64`. JAX downgrades `float64` to `float32` with a warning by default, and MLX raises an error for `float64` on GPU.
- **Adding in `uint8`.** Computing `a + b` directly wraps sums above 255.
- **Clamping after the wrap.** `torch.clamp(a + b, max=255)` returns `a + b` unchanged because the `uint8` sum has already wrapped into $[0, 255]$.
- **Forgetting to cast back to `uint8`.** Leaving the clamped result in `int16` doubles memory usage and violates the output dtype contract expected by downstream `uint8` buffers and operations.
- **Using truncating division.** `torch.div(pos, size, rounding_mode="trunc")`, `int(pos / size)`, and casting a float quotient to int all round toward zero. That makes block `0` span `-(size - 1)` through `size - 1`, nearly twice as many positions as every other block.
- **Using `fmod` for the offset.** `fmod` keeps the sign of `pos`, so negative positions produce negative offsets that silently index from the end of a buffer.
- **Dividing in `float32`.** `torch.floor(pos / size)` rounds in the right direction, but `float32` division is inexact for large integers. Because `float32` rounds `999999999` to `1000000000`, `size = 1000` puts that position in block `1000000` instead of `999999`.

## Complexity

Each function makes one pass over an input of $n$ elements in $O(n)$ time. The `int16` intermediate in `saturating_add_u8` temporarily uses twice the memory of one input tensor.

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

All three frameworks agree on the promotion rules needed for this problem. A Python int never widens a tensor, and operations on equal types preserve that type. Three cross-framework differences are worth noting. First, PyTorch treats a 0D tensor like a Python scalar during promotion, so subtracting a 0D `int32` zero point still leaves `q - zero_point` in `int8`, whereas JAX and MLX promote to `int32`. Second, MLX rejects a Python int that does not fit in the tensor's dtype, turning a silent wrap into a `ValueError`. Third, casting an out-of-range float to an integer wraps in PyTorch but saturates in JAX and MLX on this machine, so always clamp while still in floating point.
