Encode a weight matrix in MXFP4, the 4-bit format of the OCP Microscaling
specification: every block of 32 values shares one power-of-two scale, stored in a
byte, and each value is a 4-bit floating-point number with a sign, two exponent bits,
and one mantissa bit. Then decode it. Unlike the integer formats of this track, the
grid inside a block is not uniform, the scale is never anything but a power of two,
and every rounding decision is exact, so ties are real and are tested.

## Specification

```python
def mxfp4_encode(x): ...  # -> (codes, exps)
def mxfp4_decode(codes, exps): ...  # -> float32 values
```

`x` has shape `(out, in)` with `in` a multiple of 32. Each row is cut into
`in / 32` consecutive blocks of 32 values; block $b$ of row $i$ holds $x_{ij}$ for
$32 b \le j < 32 (b + 1)$.

**The element: FP4 (E2M1).** A 4-bit code has, from the highest bit to the lowest, a
sign bit $S$, a two-bit exponent field $E$, and a mantissa bit $M$:
$\text{code} = 8 S + 2 E + M$. The exponent bias is 1, and the code stands for

$$
P = (-1)^S \cdot \begin{cases}
2^{\,E - 1} \cdot (1 + M / 2) & E > 0 \quad \text{(normal)}, \\
M / 2 & E = 0 \quad \text{(subnormal, or zero)}.
\end{cases}
$$

| $2E + M$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| magnitude $\lvert P \rvert$ | $0$ | $0.5$ | $1$ | $1.5$ | $2$ | $3$ | $4$ | $6$ |

There are no encodings for infinity or NaN. Codes 0 and 8 are $+0$ and $-0$.

**The scale: E8M0.** A block's scale is $X = 2^{e}$ for an integer $e$, stored as the
byte $e + 127$ (an exponent with bias 127 and no mantissa). The value that a code
stands for in its block is $X \cdot P$.

**`mxfp4_encode(x)`**, for every block separately:

1. **The shared exponent.** Let $m = \max_j |x_j|$ over the block. $X$ is the largest
   power of two that is not above $m$, divided by the largest power of two an element
   can hold, $2^2 = 4$:
   $$
   e = \lfloor \log_2 m \rfloor - 2 , \qquad X = 2^{e} .
   $$
   A block of zeros has no such power; it gets $e = 0$.
2. **The elements.** Each $x_j$ is divided by $X$ and rounded to the nearest FP4
   value. A magnitude $|x_j| / X$ above $6$ **saturates** at $6$ (this happens: the
   division leaves the block maximum in $[4, 8)$). A value exactly halfway between two
   FP4 values goes to the one whose code has **mantissa bit 0** (round half to even):

   | $\lvert x_j \rvert / X$ | $0.25$ | $0.75$ | $1.25$ | $1.75$ | $2.5$ | $3.5$ | $5$ |
   |---|---|---|---|---|---|---|---|
   | rounds to | $0$ | $1$ | $1$ | $2$ | $2$ | $4$ | $4$ |

3. **The sign.** $S = 1$ exactly when $x_j < 0$, also when the magnitude rounds to
   zero: a small negative value is encoded as $-0$, code 8.

It returns the pair `(codes, exps)`: `codes` is a **uint8** tensor of shape
`(out, in)` with values 0 to 15, and `exps` is a **uint8** tensor of shape
`(out, in / 32)` holding the byte $e + 127$ of every block.

**`mxfp4_decode(codes, exps)`** returns the float32 tensor of shape `(out, in)` with

$$
\hat x_{ij} = 2^{\,\text{exps}_{ib} - 127} \cdot P(\text{code}_{ij}), \qquad b = \lfloor j / 32 \rfloor .
$$

`codes` and `exps` are uint8 tensors shaped as `mxfp4_encode` returns them; they need
not come from `mxfp4_encode`.

> [!WARNING]
> Dividing a float32 by an exact power of two is exact, so the ties above are decided
> by your rounding, not by noise. That requires $X$ to *be* a power of two, and
> $\lfloor \log_2 m \rfloor$ to be right when $m$ is a power of two or one ulp below
> one. A float32 `log2` and a float32 `pow` are not exact in every framework; the
> tests cover both cases.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(out, in)`, `in` a multiple of 32 | float32 |
| `codes` | `(out, in)` | uint8 |
| `exps` | `(out, in / 32)` | uint8 |
| result of `mxfp4_decode` | `(out, in)` | float32 |

## Example

One row of two blocks; only the leading values of each block are non-zero.

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

**Block 0.** $m = 7$, the largest power of two not above it is $4 = 2^2$, so $e = 0$,
$X = 1$, and the byte is $127$.

| $x$ | $5.0$ | $-3.2$ | $0.3$ | $0.75$ | $1.0$ | $-0.2$ | $2.5$ | $7.0$ |
|---|---|---|---|---|---|---|---|---|
| nearest FP4 value | $4$ (tie) | $-3$ | $0.5$ | $1$ (tie) | $1$ | $-0$ | $2$ (tie) | $6$ (saturated) |
| code | $6$ | $8 + 5$ | $1$ | $2$ | $2$ | $8 + 0$ | $4$ | $7$ |

$5.0$ is halfway between $4$ (code 6) and $6$ (code 7) and goes to the even code.
$-0.2$ is closer to $0$ than to $0.5$ and keeps its sign bit. The block maximum
itself, $7.0$, is beyond the largest element value and comes back as $6.0$.

**Block 1.** $m = 0.11$ lies between $2^{-4} = 0.0625$ and $2^{-3}$, so
$e = -4 - 2 = -6$, $X = 1/64$, and the byte is $121$. Then $0.11 \cdot 64 = 7.04$
saturates at $6$; $-0.05 \cdot 64 = -3.2$ rounds to $-3$; $0.02 \cdot 64 = 1.28$
rounds to $1.5$; and $0.004 \cdot 64 = 0.256$ rounds to the subnormal $0.5$.

## Constraints

- $1 \le \text{out} \le 64$, $32 \le \text{in} \le 512$. Every non-zero entry of `x`
  has $10^{-18} \le |x_{ij}| \le 10^{18}$, so every scale byte lies between 60 and 190;
  the same holds for the `exps` given to `mxfp4_decode`. `x` contains no $-0.0$.
- `codes` and `exps` are compared **exactly** and must have dtype uint8. Tests include
  values exactly halfway between two FP4 values, block maxima that are exact powers
  of two, and block maxima one ulp below a power of two.
- Decoded values are compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-30`.
- No Python loops over elements, blocks, or rows are needed.

## Why this matters

MXFP4 is the 4-bit format that hardware vendors agreed on: the OCP Microscaling
Formats (MX) Specification v1.0 was written by AMD, Arm, Intel, Meta, Microsoft,
NVIDIA, and Qualcomm so that accelerators could implement narrow dot products
natively. The scale is a bare exponent, so applying it is a shift, never a multiply,
and the elements are tiny floats, so a block keeps relative precision over a range of
$12 : 1$ instead of spending a uniform grid on it; storage is $4 + 8/32 = 4.25$ bits
per weight. MLX reads and writes it (`mx.quantize(w, mode="mxfp4")`), and
llm-compressor has `MXFP4` presets. Because the format is a standard, an encoder
that is slightly off produces data that every conforming decoder reads without
complaint: an exponent off by one costs a block a bit of precision or clips it, a
missing saturation turns the largest weight of a block into zero, and a decoder that
forgets the subnormal row turns every zero into half a step.
