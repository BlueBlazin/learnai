Convert between bfloat16 bit patterns and float32 values by hand: decode a uint16
tensor into the float32 values it stands for, and encode float32 values as uint16 with
round-to-nearest-even. Decoding is one shift. Encoding is where the work is, because
dropping 16 bits correctly means handling ties, a carry that runs into the exponent,
and NaN.

## Specification

```python
def bf16_to_f32(bits): ...  # uint16 -> float32
def f32_to_bf16(x): ...  # float32 -> uint16
```

A bfloat16 is the **top 16 bits of a float32**. Both formats start with the same sign
bit and the same 8-bit exponent (bias 127). bfloat16 keeps the first 7 of the 23
mantissa bits:

| bits of the uint16 | 15 | 14 to 7 | 6 to 0 |
|---|---|---|---|
| field | sign | exponent (8 bits, bias 127) | mantissa (7 bits) |

Zeros, subnormals, infinities, and NaN are encoded as in float32: exponent field 0 is
zero or a subnormal, exponent field 255 is infinity (mantissa 0) or NaN (mantissa not
0).

**`bf16_to_f32(bits)`** returns, for each 16-bit pattern $b$, the float32 whose 32-bit
pattern is $b \cdot 2^{16}$: the 16 stored bits followed by 16 zero bits. This is exact
for every pattern. Subnormals stay subnormal, $-0.0$ stays $-0.0$, and a NaN keeps its
sign and its 7 payload bits.

**`f32_to_bf16(x)`** returns the bfloat16 nearest to each element of `x`. Let $u$ be
the 32-bit pattern of the element as an unsigned integer, $h = \lfloor u / 2^{16}
\rfloor$ the 16 bits that are kept, and $r = u \bmod 2^{16}$ the 16 bits that are
dropped. Then

$$
\operatorname{f32\_to\_bf16}(x) =
\begin{cases}
\texttt{0x7FC0} + \texttt{0x8000} \cdot \text{sign} & x \text{ is NaN}, \\
h & r < \texttt{0x8000}, \\
h + 1 & r > \texttt{0x8000}, \\
h & r = \texttt{0x8000} \text{ and } h \text{ is even}, \\
h + 1 & r = \texttt{0x8000} \text{ and } h \text{ is odd}.
\end{cases}
$$

- $h + 1$ is ordinary integer addition on the whole 16-bit pattern. When the mantissa
  field is all ones, the carry increments the exponent field, which is the correct
  result: the next larger magnitude. For the largest finite magnitudes it produces the
  pattern of infinity (`0x7F80` or `0xFF80`), which is also correct: a value at least
  halfway past the largest bfloat16 rounds to infinity.
- A tie ($r$ exactly `0x8000`, the value exactly halfway between two neighbors) goes
  to the neighbor whose last mantissa bit is 0.
- NaN (exponent field 255, mantissa field not 0, in the float32 pattern) is the one
  case that is not rounded. Every NaN becomes the quiet NaN `0x7FC0`, or `0xFFC0` when
  the sign bit of the input is set. The payload is discarded.
- Infinities, zeros of either sign, and subnormals need no special treatment: the rule
  above is already right for them.

Work on the bit patterns with integer operations (shifts, masks, additions,
comparisons). The only floating-point step in each function is reinterpreting 32 bits
between float32 and a 32-bit integer type, which every framework can do without
changing a bit. Do not use your framework's `bfloat16` dtype in any spelling
(`torch.bfloat16`, `Tensor.bfloat16()`, `jnp.bfloat16`, `mx.bfloat16`, `"bfloat16"`).

> [!NOTE]
> PyTorch can store and convert uint16 tensors but implements almost no arithmetic
> for them (no shifts, no addition). Convert to a wider integer type before computing,
> and to uint16 only at the end.

## Shapes

| name | shape | dtype |
|---|---|---|
| `bits` | any shape with one to three dimensions | uint16 |
| result of `bf16_to_f32` | same as `bits` | float32 |
| `x` | any shape with one to three dimensions | float32 |
| result of `f32_to_bf16` | same as `x` | uint16 |

## Example

```python
bf16_to_f32([0x3F80, 0xC020, 0x4049, 0x7F80, 0x0001])
# [1.0, -2.5, 3.140625, inf, 9.18355e-41]

f32_to_bf16([1.0, -2.5, 3.14159265, 0.1, 1.00390625, 1.01171875])
# [0x3F80, 0xC020, 0x4049, 0x3DCD, 0x3F80, 0x3F82]
```

Decoding `0x4049`: the sign bit is 0, the exponent field is `0x80` $= 128$, and the
mantissa field is `0x49` $= 73$, so the value is $2^{128 - 127} \cdot (1 + 73/128) =
3.140625$. The float32 with pattern `0x40490000` is the same number.

Encoding, element by element:

| $x$ | float32 pattern | kept $h$ | dropped $r$ | rule | result | decodes to |
|---|---|---|---|---|---|---|
| $1.0$ | `0x3F800000` | `0x3F80` | `0x0000` | exact | `0x3F80` | $1.0$ |
| $-2.5$ | `0xC0200000` | `0xC020` | `0x0000` | exact | `0xC020` | $-2.5$ |
| $3.14159265$ | `0x40490FDB` | `0x4049` | `0x0FDB` | below half: $h$ | `0x4049` | $3.140625$ |
| $0.1$ | `0x3DCCCCCD` | `0x3DCC` | `0xCCCD` | above half: $h + 1$ | `0x3DCD` | $0.100097656$ |
| $1.00390625$ | `0x3F808000` | `0x3F80` | `0x8000` | tie, $h$ even: $h$ | `0x3F80` | $1.0$ |
| $1.01171875$ | `0x3F818000` | `0x3F81` | `0x8000` | tie, $h$ odd: $h + 1$ | `0x3F82` | $1.015625$ |

For $0.1$, truncation would keep `0x3DCC`, which decodes to $0.099609375$: an error of
$3.9 \cdot 10^{-4}$ where rounding makes one of $9.8 \cdot 10^{-5}$.

## Constraints

- Every comparison is **exact**, bit for bit: the uint16 patterns from `f32_to_bf16`,
  and the 32-bit patterns of the floats from `bf16_to_f32` (so the sign of zero and
  the bits of a NaN count).
- `bits` may hold any of the 65,536 patterns. `x` may hold any float32, including
  subnormals, infinities, and NaN with any payload and either sign.
- `f32_to_bf16` must return dtype uint16, and `bf16_to_f32` dtype float32.
- Inputs have at most 100,000 elements. No Python loops over elements are needed.

## Why this matters

Most open LLM checkpoints are stored in bfloat16: it halves the file, and because it
keeps float32's exponent it covers the same range, where IEEE half precision overflows
at 65,504. A loader that reads those two bytes as half precision produces weights that
can be wrong by orders of magnitude, with no error raised. The reverse direction runs
whenever weights or activations are cast down for inference. Truncating there biases
every value toward zero by up to a full step, a NaN that is rounded like a number can
turn into infinity or zero and hide the fault that produced it, and a hand-rolled
rounding that mishandles the carry halves weights that sit just below a power of two.
