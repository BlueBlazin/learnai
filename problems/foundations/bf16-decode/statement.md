Implement `bf16_to_f32` and `f32_to_bf16` to convert between bfloat16 bit patterns (`uint16`) and `float32` values using integer bit operations. `bf16_to_f32` decodes 16-bit patterns into exact `float32` values, and `f32_to_bf16` encodes `float32` values into `uint16` patterns with round-to-nearest-even.

## Specification

```python
def bf16_to_f32(bits): ...  # uint16 -> float32
def f32_to_bf16(x): ...  # float32 -> uint16
```

A bfloat16 stores the **top 16 bits of a float32**. Both formats use the same sign bit and the same 8-bit exponent (bias 127), while bfloat16 keeps only the first 7 of the 23 mantissa bits.

| bits of the uint16 | 15 | 14 to 7 | 6 to 0 |
|---|---|---|---|
| field | sign | exponent (8 bits, bias 127) | mantissa (7 bits) |

![The 32 bits of $0.1$ in float32: bfloat16 keeps the top 16 ($h$), and the 16 dropped bits ($r$) decide the rounding.](assets/bits.svg)

Zeros, subnormals, infinities, and NaN follow the same bit conventions as float32. An exponent field of 0 represents zero (mantissa 0) or a subnormal (nonzero mantissa), and an exponent field of 255 represents infinity (mantissa 0) or NaN (nonzero mantissa).

**`bf16_to_f32(bits)`** maps each 16-bit pattern $b$ to the float32 with 32-bit pattern $b \cdot 2^{16}$, appending 16 zero bits after the stored bits. Every pattern decodes exactly. Subnormals remain subnormal, $-0.0$ remains $-0.0$, and each NaN keeps its sign and 7 payload bits.

**`f32_to_bf16(x)`** maps each float32 element of `x` to its nearest bfloat16 pattern. Let $u$ be the element's 32-bit pattern as an unsigned integer, $\text{sign} \in \{0, 1\}$ be its top bit, $h = \lfloor u / 2^{16} \rfloor$ be the 16 kept bits, and $r = u \bmod 2^{16}$ be the 16 dropped bits. Then

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

- $h + 1$ is ordinary integer addition on the full 16-bit pattern. When the 7-bit mantissa is all ones, the carry increments the exponent field to produce the next larger representable magnitude. For the largest finite magnitudes, the carry produces infinity (`0x7F80` or `0xFF80`), which is the correct result when a value sits at least halfway past the largest finite bfloat16.
- A tie ($r = \texttt{0x8000}$, halfway between two neighboring bfloat16 values) rounds to the neighbor whose least significant mantissa bit is 0.
- NaN (exponent field 255 and nonzero mantissa in the float32 pattern) is not rounded. Every NaN becomes the quiet NaN `0x7FC0`, or `0xFFC0` if the input's sign bit is set, discarding the payload.
- Infinities, signed zeros, and subnormals need no special handling because the rule above already produces the right pattern for them.

Operate on the bit patterns using integer operations (shifts, masks, additions, and comparisons). The only floating-point step in each function is reinterpreting 32 bits between `float32` and a 32-bit integer type without altering any bits. Do not use your framework's `bfloat16` dtype in any spelling (`torch.bfloat16`, `Tensor.bfloat16()`, `jnp.bfloat16`, `mx.bfloat16`, `"bfloat16"`).

> [!NOTE]
> PyTorch can store and convert `uint16` tensors but supports almost no arithmetic on them (no shifts or addition). Convert to a wider integer type before computing, and convert back to `uint16` at the end.

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

To decode `0x4049`, read its sign bit (0), exponent field (`0x80` $= 128$), and mantissa field (`0x49` $= 73$). The value is $2^{128 - 127} \cdot (1 + 73/128) = 3.140625$, which matches the float32 bit pattern `0x40490000`.

Encoding element by element:

| $x$ | float32 pattern | kept $h$ | dropped $r$ | rule | result | decodes to |
|---|---|---|---|---|---|---|
| $1.0$ | `0x3F800000` | `0x3F80` | `0x0000` | exact | `0x3F80` | $1.0$ |
| $-2.5$ | `0xC0200000` | `0xC020` | `0x0000` | exact | `0xC020` | $-2.5$ |
| $3.14159265$ | `0x40490FDB` | `0x4049` | `0x0FDB` | below half: $h$ | `0x4049` | $3.140625$ |
| $0.1$ | `0x3DCCCCCD` | `0x3DCC` | `0xCCCD` | above half: $h + 1$ | `0x3DCD` | $0.100097656$ |
| $1.00390625$ | `0x3F808000` | `0x3F80` | `0x8000` | tie, $h$ even: $h$ | `0x3F80` | $1.0$ |
| $1.01171875$ | `0x3F818000` | `0x3F81` | `0x8000` | tie, $h$ odd: $h + 1$ | `0x3F82` | $1.015625$ |

For $0.1$, simple truncation would keep `0x3DCC` ($0.099609375$), an error of $3.9 \cdot 10^{-4}$ instead of $9.8 \cdot 10^{-5}$ with rounding.

## Constraints

- Every comparison is **exact**, bit for bit, on both the `uint16` patterns from `f32_to_bf16` and the 32-bit patterns of the floats from `bf16_to_f32` (including the sign of zero and all bits of a NaN).
- `bits` may contain any of the 65,536 possible 16-bit patterns. `x` may contain any float32 value, including subnormals, infinities, and NaN with any payload and either sign.
- `f32_to_bf16` must return dtype `uint16`, and `bf16_to_f32` must return dtype `float32`.
- Inputs have at most 100,000 elements. No Python loops over elements are needed.

## Why this matters

Most open LLM checkpoints are stored in bfloat16 because it halves file size while keeping float32's dynamic range, avoiding the 65,504 overflow limit of IEEE float16. A checkpoint loader that misinterprets those two bytes as float16 produces weights that are off by orders of magnitude without raising any error.

The reverse conversion runs whenever weights or activations are cast down for inference or storage. Truncating instead of rounding biases every number toward zero by up to a full step. Rounding a NaN like an ordinary number can turn it into infinity or zero and mask the fault that created it. Mishandling the carry out of an all-ones mantissa can halve weights sitting just below a power of two.
