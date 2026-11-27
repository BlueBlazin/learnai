Write three small integer functions: turn int8 codes back into real numbers, add two
uint8 tensors without wrapping, and split a position that may be negative into a
block number and an offset. Each formula is one line. Each has a version that looks
the same, raises nothing, and is wrong on part of the input range, because of the
type the arithmetic runs in or the way a division rounds.

## Specification

```python
def dequantize(q, scale, zero_point): ...
def saturating_add_u8(a, b): ...
def block_and_offset(pos, size): ...
```

**`dequantize(q, scale, zero_point)`** takes int8 codes `q` and returns, as float32
in the shape of `q`,

$$
\text{out} = (q - \text{zero\_point}) \cdot \text{scale} ,
$$

where $q - \text{zero\_point}$ is the difference of the two integers, a whole number
between $-255$ and $255$.

**`saturating_add_u8(a, b)`** takes two uint8 tensors of one shape and returns, as
**uint8** in that shape,

$$
\text{out} = \min(a + b,\ 255) ,
$$

where $a + b$ is the sum of the two integers, between 0 and 510.

**`block_and_offset(pos, size)`** takes integer positions `pos`, which may be
negative, and returns the tuple `(block, offset)` of two integer tensors in the shape
of `pos`:

$$
\text{block} = \left\lfloor \frac{\text{pos}}{\text{size}} \right\rfloor , \qquad
\text{offset} = \text{pos} - \text{block} \cdot \text{size} .
$$

The division rounds toward negative infinity, so $0 \le \text{offset} < \text{size}$
for every position, and position $-1$ is the last entry of block $-1$.

- `q`: an int8 tensor. `scale`: a positive Python float. `zero_point`: a Python int
  with $-128 \le \text{zero\_point} \le 127$.
- `a`, `b`: uint8 tensors.
- `pos`: an integer tensor of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX). `size`: a positive Python int.
- Dtypes of the results. `dequantize` returns float32. `saturating_add_u8` returns
  uint8, and a correct sum in a wider integer type is rejected. `block` and `offset`
  are integer tensors, and any integer width is accepted for them.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | any, one to three axes | int8 |
| result of `dequantize` | the shape of `q` | float32 |
| `a`, `b` | any, one to three axes, equal | uint8 |
| result of `saturating_add_u8` | the shape of `a` | uint8 |
| `pos` | any, one to three axes | integer |
| `block`, `offset` | the shape of `pos` | integer |

## Tools

- **The result type of an operation** is decided by the types of its operands, never
  by the values. Two tensors of one integer type give that type. Two different types
  give a type that holds both: uint8 with int8 gives int16, and an integer with a
  float gives a float.
- **A Python number does not widen a tensor.** A Python int that fits the type of
  an integer tensor is combined with it in that type, in all three frameworks. One
  that does not fit (`200` with int8) is wrapped to that type in PyTorch and JAX,
  and MLX raises `ValueError`.
- **Integer arithmetic wraps.** A result outside the range of its type is reduced
  modulo $2^{\text{bits}}$, with no error and no warning. Casting to a narrower
  integer type wraps the same way.
- **Casting** is `x.to(torch.int16)` in PyTorch and `x.astype(jnp.int16)`,
  `x.astype(mx.int16)` in JAX and MLX.
- **Integer division has two conventions.** Floor division rounds toward negative
  infinity, and its remainder has the sign of the divisor. Truncating division
  rounds toward zero, and its remainder has the sign of the dividend. They agree
  whenever both operands are positive. The floor pair is `floor_divide` and
  `remainder` in all three frameworks. The truncating pair is
  `torch.div(..., rounding_mode="trunc")` and `torch.fmod` in PyTorch and
  `jax.lax.div` and `jnp.fmod` in JAX; MLX has neither. In all three, casting a
  float to an integer truncates.

> [!WARNING]
> A check made after the overflow sees only the wrapped value, and the wrapped value
> is in range.

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

Computed in int8, `100 - (-100)` is `-56`, and the third value comes out as `-28.0`.
Computed in uint8, `200 + 100` is `44` and `255 + 255` is `254`. With truncating
division, `-7 / 3` is `-2`, and the remainder that goes with it is `-1`.

## Constraints

- `saturating_add_u8` and `block_and_offset` are compared with the reference
  exactly. `dequantize` is compared to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`, which also accepts `q * scale - zero_point * scale`
  computed in float32.
- Tensors have one to three axes and at most 256 entries. An axis can have size 1.
- `q` takes every int8 value and `a`, `b` every uint8 value. `scale` is between
  `0.001` and `1`.
- $|\text{pos}| \le 10^9$ and $1 \le \text{size} \le 10^6$.
- No Python loops are needed.

## Why this matters

Affine int8 Quantization ends with this `dequantize`, and An Integer-Only Linear
Layer widens its int8 operands to int32 before the matrix product for the same
reason: the result type is int8 otherwise. Packing 4-bit Codes shifts uint8 codes
into 32-bit words, and a uint8 shifted left stays uint8 and loses its high bits.
The wrapped values are ordinary numbers of the right shape and type. In a quantized
model they show up as a quality loss on some inputs, the ones whose codes sit far
from the zero point, and tests with small values pass.
