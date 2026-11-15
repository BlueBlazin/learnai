## The idea

A microscaling block is a small floating-point number system with the exponent split
in two. The high part of the exponent, the scale $X = 2^e$, is shared by 32 values and
costs 8 bits once. The low part, two bits, stays with each element next to one
mantissa bit and a sign. Within a block the representable magnitudes are
$X \cdot \{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$: steps of $0.5X$ up to $2X$, then $X$, then
$2X$. Small values of the block get fine steps and large ones coarse steps, which is
what a float does and a uniform 4-bit grid does not.

**Choosing the scale.** The specification's conversion (section 6.3) sets $X$ to the
largest power of two not above the block maximum $m$, divided by the largest power of
two an element can represent, $2^{e_{\max}} = 4$. So $m / X$ lies in $[4, 8)$: the
maximum always lands in the top binade of the element. The top binade only has the
values $4$ and $6$, so a maximum in $(6, 8) \cdot X$ is above the largest element and
is clamped to $6X$, as the same section prescribes. This is deliberate: a scale twice
as large would represent the maximum without clamping and cost every other element of
the block one bit of precision.

**The exponent is already in the float.** A float32 is stored as sign, an 8-bit
exponent field with bias 127, and 23 mantissa bits. The E8M0 scale is, in the
specification's words, an unsigned representation of a conventional biased float32
exponent (section 5.4.1). So no logarithm is needed in either direction:

- $\lfloor \log_2 m \rfloor + 127$ is the exponent field of $m$: reinterpret the bits
  as an int32 and shift right by 23. The scale byte is that field minus 2.
- $2^{\text{byte} - 127}$ is the float32 whose exponent field is the byte and whose
  mantissa is zero: shift the byte left by 23 and reinterpret.

Both are exact. The obvious alternatives are not: in float32, `log2` of the largest
number below $2^{20}$ is $20 - 8.6 \cdot 10^{-8}$, which rounds to $20.0$, and its
floor is off by one; and `pow(2.0, e)` is computed through logarithms on some
backends and misses the power of two by an ulp (measured: `mx.power(2.0, e)` in MLX
0.32.4 for negative integers $e$, and `jnp.exp2`).

**Rounding.** Because $X$ is a power of two, $x / X$ is exact, and the only rounding
is the one onto the element grid. The grid is uniform within each of three regions,
so rounding to nearest there is an integer rounding in units of the local step, and
the rounded count maps to the code by an offset:

$$
2E + M = \begin{cases}
\operatorname{round}(2a) & a < 2, \\
\operatorname{round}(a) + 2 & 2 \le a < 4, \\
\operatorname{round}(a / 2) + 4 & 4 \le a \le 6 .
\end{cases}
$$

The offsets are even, so the parity of the count is the mantissa bit, and the
frameworks' round-half-to-even is exactly IEEE 754's roundTiesToEven on this format,
which the specification requires implementations to support (section 5.3.3).

## Reference solution (PyTorch)

```python
BLOCK, EMAX, BIAS = 32, 2, 127
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _exponent_field(m):
    return m.view(torch.int32) >> 23


def _power_of_two(field):
    return (field << 23).view(torch.float32)


def mxfp4_encode(x):
    out_dim, in_dim = x.shape
    blocks = x.reshape(out_dim, in_dim // BLOCK, BLOCK)
    largest = blocks.abs().amax(dim=-1)
    field = torch.where(largest > 0, _exponent_field(largest) - EMAX, BIAS)
    scale = _power_of_two(field)
    a = (blocks.abs() / scale[..., None]).clamp(max=6.0)
    magnitude = torch.where(
        a < 2.0,
        torch.round(a * 2.0),
        torch.where(a < 4.0, torch.round(a) + 2.0, torch.round(a / 2.0) + 4.0),
    )
    codes = magnitude + 8.0 * (blocks < 0)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), field.to(torch.uint8)


def mxfp4_decode(codes, exps):
    codes = codes.to(torch.int32)
    magnitude = torch.tensor(MAGNITUDES)[codes & 7]
    sign = 1.0 - 2.0 * (codes >> 3)
    scale = _power_of_two(exps.to(torch.int32))
    return sign * magnitude * scale.repeat_interleave(BLOCK, dim=1)
```

The JAX and MLX versions are line-for-line translations; `view` reinterprets bits in
all three. Where `frexp` and `ldexp` exist (PyTorch, JAX) they are an equally exact
way to write the two helper functions.

The decoder's table is the element formula evaluated for the eight values of
$2E + M$. The first two entries are the subnormal row, $E = 0$: the implicit leading
one is absent, and the exponent is that of $E = 1$.

## Common mistakes

- **An exponent per element.** Normalizing every element by its own power of two
  encodes each as if it were the block maximum. The result decodes to the right order
  of magnitude only for the largest element; the point of the format is that the
  exponent is shared.
- **The maximum over the wrong extent.** A scale per row, instead of per block of 32,
  gives every block the scale of the largest block in its row.
- **Forgetting $e_{\max}$.** With $e = \lfloor \log_2 m \rfloor$ the maximum lands in
  $[1, 2)$, and the codes for $2$, $3$, $4$, and $6$ are never used: a 3-bit format.
- **A scale that avoids saturation.** Picking the smallest power of two with
  $m / X \le 6$ seems safer, and is what MLX does: `mx.quantize(w, mode="mxfp4")` in
  MLX 0.32.4 returns a scale byte one higher than the specification's rule whenever
  the maximum would be clamped (measured while writing this problem; its element
  codes and its decoder agree with this problem bit for bit). Section 6.3 allows
  other algorithms, so that is conforming, but it is not the algorithm specified
  here.
- **`floor(log2(m))` in float32.** Right for almost every input, and one too large
  for a maximum just below a large power of two.
- **Truncating.** Rounding toward zero doubles the worst-case error and biases every
  weight toward zero.
- **Ties away from zero.** `floor(v + 0.5)` sends $2.5$ to $3$ and $5$ to $6$; the
  format's rounding sends them to $2$ and $4$, the codes with mantissa bit 0.
- **No saturation.** Without the clamp, a magnitude of $7$ or more rounds to $8$,
  which would be code 8: the sign bit. The largest weight of the block decodes to
  $-0$.
- **Ignoring subnormals in the encoder.** Treating the grid as starting at $1$ rounds
  everything below $1$ to $0$ or $1$ and never uses code 1, the value $0.5$.
- **Ignoring subnormals in the decoder.** Applying $2^{E-1}(1 + M/2)$ to $E = 0$
  gives $0.5$ and $0.75$ for codes 0 and 1. Every zero weight then decodes to half a
  step.
- **Dropping the sign of a zero.** IEEE rounding keeps the sign of a value that
  rounds to zero, and so does this format: code 8 is $-0$. A decoder never sees the
  difference; a byte-exact comparison of two encoders does.
- **Arithmetic on the scale byte in uint8.** `exps - 127` wraps around for every
  byte below 127, that is, for every block with a scale below 1, which is most
  blocks of a real weight matrix. Widen first.

## Complexity

Encoding is one pass for the block maxima and one for the elements; decoding is a
lookup and a multiplication per element. Storage is $4 + 8 / 32 = 4.25$ bits per
weight.

## In the wild

The format is defined by the *OCP Microscaling Formats (MX) Specification*, version
1.0 (Open Compute Project, September 2023): the concrete formats in section 5.2
(MXFP4 is FP4 elements, block size 32, an E8M0 scale), the FP4 encoding and its
rounding and saturation rules in section 5.3.3, the E8M0 scale in section 5.4.1, and
the conversion from a vector of scalars in section 6.3. The scale byte 255 is
reserved for NaN, and FP4 has no infinities or NaNs. The specification does not
prescribe how a block is laid out in memory.

In MLX, `mx.quantize(w, mode="mxfp4")` packs the 4-bit codes eight to a uint32, lowest
index in the lowest bits, the layout of `bit-packing`, and returns the scale bytes as
uint8; `mx.dequantize(..., mode="mxfp4")` is this problem's decoder. llm-compressor
lists `MXFP4` among its preset schemes.
