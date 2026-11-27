Implement encoding and decoding for MXFP4, the 4-bit microscaling format from the OCP Microscaling specification. Each block of 32 weights shares a single power-of-two scale stored in one byte, and each weight is stored as a 4-bit floating-point number with 1 sign bit, 2 exponent bits, and 1 mantissa bit. Because the scale is always an exact power of two, dividing by it introduces no rounding error, so exact ties between adjacent FP4 levels occur and are tested.

## Specification

```python
def mxfp4_encode(x): ...  # -> (codes, exps)
def mxfp4_decode(codes, exps): ...  # -> float32 values
```

The input `x` has shape `(out, in)` where `in` is a multiple of 32. Each row is split into `in / 32` consecutive blocks of 32 elements, so block $b$ of row $i$ contains $x_{ij}$ for $32b \le j < 32(b + 1)$.

**Element format: FP4 (E2M1).** A 4-bit code packs a sign bit $S$, a 2-bit exponent field $E$, and a 1-bit mantissa $M$ from most significant to least significant bit, giving $\text{code} = 8S + 2E + M$. With an exponent bias of 1, the code represents the value

$$
P = (-1)^S \cdot \begin{cases}
2^{\,E - 1} \cdot (1 + M / 2) & E > 0 \quad \text{(normal)}, \\
M / 2 & E = 0 \quad \text{(subnormal, or zero)}.
\end{cases}
$$

| $2E + M$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| magnitude $\lvert P \rvert$ | $0$ | $0.5$ | $1$ | $1.5$ | $2$ | $3$ | $4$ | $6$ |

FP4 has no infinity or NaN encodings. Codes 0 and 8 represent $+0$ and $-0$.

**Scale format: E8M0.** Each block has a power-of-two scale $X = 2^e$ for an integer exponent $e$, stored as the unsigned byte $e + 127$ (an 8-bit biased exponent with no mantissa). Each encoded element in the block represents $X \cdot P$.

**`mxfp4_encode(x)`** processes every block independently in three steps:

1. **Shared exponent.** Let $m = \max_j |x_j|$ be the largest magnitude in the block. Set $X$ to the largest power of two not exceeding $m$, divided by the largest power of two representable by an FP4 element ($2^2 = 4$):
   $$
   e = \lfloor \log_2 m \rfloor - 2 , \qquad X = 2^{e} .
   $$
   If every element in the block is zero, set $e = 0$.
2. **Element rounding.** Divide each $x_j$ by $X$ and round to the nearest FP4 value. Magnitudes $|x_j| / X$ above $6$ **saturate** at $6$, which happens whenever the block maximum falls in $(6X, 8X)$. A value halfway between two FP4 levels rounds to the level whose code has **mantissa bit 0** (round half to even):

   | $\lvert x_j \rvert / X$ | $0.25$ | $0.75$ | $1.25$ | $1.75$ | $2.5$ | $3.5$ | $5$ |
   |---|---|---|---|---|---|---|---|
   | rounds to | $0$ | $1$ | $1$ | $2$ | $2$ | $4$ | $4$ |

3. **Sign bit.** Set $S = 1$ whenever $x_j < 0$, even if the magnitude rounds to zero. A small negative value therefore encodes as $-0$ (code 8).

Return `(codes, exps)`, where `codes` holds values in $\{0, \dots, 15\}$ and `exps` holds the byte $e + 127$ for each block.

**`mxfp4_decode(codes, exps)`** reconstructs the float32 matrix from the block scales and element codes:

$$
\hat x_{ij} = 2^{\,\text{exps}_{ib} - 127} \cdot P(\text{code}_{ij}), \qquad b = \lfloor j / 32 \rfloor .
$$

The inputs `codes` and `exps` follow the shapes and dtypes below, and they do not need to come from `mxfp4_encode`.

> [!WARNING]
> Dividing a float32 by an exact power of two is exact, so midpoint ties are decided by your rounding rule rather than floating-point noise. This requires $X$ to be an exact power of two and $\lfloor \log_2 m \rfloor$ to be exact when $m$ is a power of two or one ulp below one. Float32 `log2` and `pow` are not exact in every framework, and the tests cover both edge cases.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(out, in)`, `in` a multiple of 32 | float32 |
| `codes` | `(out, in)` | uint8 |
| `exps` | `(out, in / 32)` | uint8 |
| result of `mxfp4_decode` | `(out, in)` | float32 |

## Example

Consider one row with two blocks where only the leading entries of each block are non-zero.

```python
x[0, 0:8] = [5.0, -3.2, 0.3, 0.75, 1.0, -0.2, 2.5, 7.0]  # x[0, 8:32] = 0
x[0, 32:36] = [0.11, -0.05, 0.02, 0.004]  # x[0, 36:64] = 0

codes, exps = mxfp4_encode(x)
# exps = [[127, 121]]
# codes[0, 0:8]   = [6, 13, 1, 2, 2, 8, 4, 7]
# codes[0, 32:36] = [7, 13, 3, 1]

mxfp4_decode(codes, exps)
# [0, 0:8]   = [4.0, -3.0, 0.5, 1.0, 1.0, -0.0, 2.0, 6.0]
# [0, 32:36] = [0.09375, -0.046875, 0.0234375, 0.0078125]
```

**Block 0.** The maximum magnitude is $m = 7$. The largest power of two not above $7$ is $4 = 2^2$, so $e = 2 - 2 = 0$, $X = 1$, and the stored scale byte is $127$.

| $x$ | $5.0$ | $-3.2$ | $0.3$ | $0.75$ | $1.0$ | $-0.2$ | $2.5$ | $7.0$ |
|---|---|---|---|---|---|---|---|---|
| nearest FP4 value | $4$ (tie) | $-3$ | $0.5$ | $1$ (tie) | $1$ | $-0$ | $2$ (tie) | $6$ (saturated) |
| code | $6$ | $8 + 5$ | $1$ | $2$ | $2$ | $8 + 0$ | $4$ | $7$ |

Here $5.0$ sits halfway between $4$ (code 6) and $6$ (code 7), so it rounds to the even code 6. The value $-0.2$ is closer to $0$ than to $0.5$ and keeps its sign bit, producing code 8. The block maximum $7.0$ exceeds the largest FP4 value and saturates to $6.0$.

**Block 1.** The maximum magnitude is $m = 0.11$, which lies between $2^{-4} = 0.0625$ and $2^{-3}$. That gives $e = -4 - 2 = -6$, $X = 1/64$, and scale byte $121$. Dividing by $X$ multiplies each value by 64. The largest entry $0.11 \cdot 64 = 7.04$ saturates at $6$, $-0.05 \cdot 64 = -3.2$ rounds to $-3$, $0.02 \cdot 64 = 1.28$ rounds to $1.5$, and $0.004 \cdot 64 = 0.256$ rounds to the subnormal $0.5$.

## Constraints

- $1 \le \text{out} \le 64$ and $32 \le \text{in} \le 512$. Every non-zero entry of `x` satisfies $10^{-18} \le |x_{ij}| \le 10^{18}$, so every scale byte lies between 60 and 190. The same range holds for the `exps` passed to `mxfp4_decode`. `x` contains no $-0.0$.
- `codes` and `exps` are compared **exactly** and must have dtype uint8. Tests include exact midpoints between FP4 values, block maxima that are exact powers of two, and block maxima one ulp below a power of two.
- Decoded values are compared with a relative tolerance of `1e-4` and an absolute tolerance of `1e-30`.
- No Python loops over elements, blocks, or rows are needed.

## Why this matters

MXFP4 is a hardware-standard 4-bit format defined in the OCP Microscaling Formats (MX) Specification v1.0 by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm so accelerators can execute narrow dot products natively. Because the shared scale is a pure power of two, applying it is an exponent addition rather than a floating-point multiply. Because the elements are tiny floats, each block preserves relative precision over a $12 : 1$ range instead of spending codes on a uniform grid, using $4 + 8/32 = 4.25$ bits per weight. Both MLX (`mx.quantize(w, mode="mxfp4")`) and llm-compressor (`MXFP4`) support it.

Because MXFP4 is an interchange standard, a subtle encoder or decoder bug produces valid bytes that downstream tools read without error. An exponent off by one wastes a bit of precision or clips the block. Missing saturation overflows the largest weight in a block into code 8 (`-0`). Forgetting the subnormal case in the decoder turns every zero weight into half a step.
