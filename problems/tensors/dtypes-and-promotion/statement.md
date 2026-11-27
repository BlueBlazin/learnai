Implement three integer-arithmetic functions that avoid silent overflow and rounding bugs. You will dequantize `int8` codes to `float32`, add two `uint8` tensors with saturation, and split possibly negative positions into block indices and non-negative offsets.

## Specification

```python
def dequantize(q, scale, zero_point): ...
def saturating_add_u8(a, b): ...
def block_and_offset(pos, size): ...
```

**`dequantize(q, scale, zero_point)`** takes an `int8` tensor `q`, a positive Python float `scale`, and a Python integer `zero_point` in $[-128, 127]$. It returns a `float32` tensor with the same shape as `q`:

$$
\text{out} = (q - \text{zero\_point}) \cdot \text{scale}.
$$

Here $q - \text{zero\_point}$ is the exact mathematical difference of the two integers, which lies in $[-255, 255]$.

**`saturating_add_u8(a, b)`** takes two `uint8` tensors `a` and `b` of the same shape and returns a **`uint8`** tensor of that shape:

$$
\text{out} = \min(a + b,\, 255).
$$

Here $a + b$ is the exact integer sum in $[0, 510]$. Returning a wider integer dtype is rejected even if the numerical values are right.

**`block_and_offset(pos, size)`** takes an integer tensor `pos` (in your framework's default integer type: `int64` in PyTorch, `int32` in JAX and MLX) and a positive Python integer `size`. Entries in `pos` may be negative. It returns a tuple `(block, offset)` of two integer tensors with the same shape as `pos` (any integer width is accepted):

$$
\text{block} = \left\lfloor \frac{\text{pos}}{\text{size}} \right\rfloor, \qquad
\text{offset} = \text{pos} - \text{block} \cdot \text{size}.
$$

Because the division rounds toward negative infinity, $0 \le \text{offset} < \text{size}$ for every position. In particular, position $-1$ maps to the last entry of block $-1$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | any, 1 to 3 axes | int8 |
| result of `dequantize` | same as `q` | float32 |
| `a`, `b` | any, 1 to 3 axes, equal | uint8 |
| result of `saturating_add_u8` | same as `a` | uint8 |
| `pos` | any, 1 to 3 axes | integer |
| `block`, `offset` | same as `pos` | integer |

## Tools

- **Result dtype promotion** depends only on the operand types, never on their values. Two tensors of the same integer type keep that type. Two different types promote to a type that holds both, so `uint8` with `int8` yields `int16`, and an integer with a float yields a float.
- **Python scalars do not widen tensors.** A Python int that fits in an integer tensor's type is combined in that tensor's type across all three frameworks. If the Python int does not fit (such as `200` with `int8`), PyTorch and JAX wrap it to that type, while MLX raises `ValueError`.
- **Integer arithmetic wraps silently.** Results outside the range of the dtype wrap modulo $2^{\text{bits}}$ with no error or warning. Casting to a narrower integer type wraps the same way.
- **Casting** uses `x.to(torch.int16)` in PyTorch, `x.astype(jnp.int16)` in JAX, and `x.astype(mx.int16)` in MLX.
- **Integer division has two conventions.** Floor division rounds toward negative infinity, and its remainder has the sign of the divisor. Truncating division rounds toward zero, and its remainder has the sign of the dividend. Both agree when operands are positive. All three frameworks provide the floor pair as `floor_divide` and `remainder`. For truncation, PyTorch provides `torch.div(..., rounding_mode="trunc")` and `torch.fmod`, JAX provides `jax.lax.div` and `jnp.fmod`, and MLX provides neither. Casting a float to an integer truncates in all three frameworks.

> [!WARNING]
> A clamp or range check applied after an overflow only sees the already-wrapped value, which is already in range.

## Example

```python
dequantize(q=[-128, 0, 100, 127], scale=0.5, zero_point=-100)  # q is int8
# [-14.0, 50.0, 100.0, 113.5]        # (100 + 100) * 0.5 = 100.0

saturating_add_u8([0, 100, 200, 255], [0, 100, 100, 255])  # both uint8
# [0, 200, 255, 255]                 # uint8; the sums are 0, 200, 300, 510

block_and_offset(pos=[-7, -6, -1, 0, 1, 6, 7], size=3)
# block:  [-3, -2, -1, 0, 0, 2, 2]
# offset: [ 2,  0,  2, 0, 1, 0, 1]   # -7 = -3 * 3 + 2
```

If you compute these naively, each one fails silently on part of the input. In `int8`, `100 - (-100)` wraps to `-56`, turning the third dequantized value into `-28.0`. In `uint8`, `200 + 100` wraps to `44` and `255 + 255` wraps to `254`. Under truncating division, `-7 / 3` gives `-2` with remainder `-1`.

## Constraints

- `saturating_add_u8` and `block_and_offset` are compared to the reference for exact equality.
- `dequantize` is compared with relative tolerance `1e-4` and absolute tolerance `1e-4` (which also accepts `q * scale - zero_point * scale` computed in `float32`).
- Tensors have 1 to 3 axes and at most 256 entries. Any axis may have size 1.
- `q` covers all `int8` values, `a` and `b` cover all `uint8` values, and `0.001 <= scale <= 1`.
- $|\text{pos}| \le 10^9$ and $1 \le \text{size} \le 10^6$.
- No Python loops are needed.

## Why this matters

Quantized inference depends on explicit widening before arithmetic. Affine `int8` dequantization uses this exact `dequantize` formula, integer-only linear layers widen `int8` inputs to `int32` before matrix multiplication so products do not overflow, and 4-bit packing widens `uint8` codes before shifting them into 32-bit words so high bits are not lost.

Wrapped integers look like ordinary numbers with the expected shape and dtype. In a quantized model, overflow shows up only as degraded quality on inputs whose codes sit far from the zero point, while unit tests with small values pass.
