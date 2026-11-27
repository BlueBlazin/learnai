## The idea

A microscaling block is a floating-point representation whose exponent is split in two. The high part is a shared scale $X = 2^e$ stored once in 8 bits for 32 elements. The low part stays with each element as a 2-bit exponent alongside 1 mantissa bit and 1 sign bit. Within a block, the representable magnitudes are $X \cdot \{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$, with step size $0.5X$ up to $2X$, $X$ up to $4X$, and $2X$ up to $6X$. Small weights get fine resolution and large weights get coarse resolution, matching floating-point arithmetic rather than a uniform integer grid.

**Choosing the scale.** Section 6.3 of the OCP specification sets $X$ to the largest power of two not exceeding the block maximum $m$, divided by the largest power of two an element can represent ($2^{e_{\max}} = 4$). This puts $m / X$ in $[4, 8)$, placing the maximum in the top binade of FP4. Because that top binade only contains $4$ and $6$, any maximum in $(6X, 8X)$ saturates to $6X$. This trade-off is deliberate. Doubling $X$ would avoid clamping the maximum, but it would cost every other element in the block one bit of precision.

**Extracting the exponent from float32 bits.** A float32 value consists of 1 sign bit, an 8-bit exponent field with bias 127, and 23 mantissa bits. The E8M0 scale is, in the specification's words (section 5.4.1), an unsigned representation of a conventional biased float32 exponent. That means no logarithm or power call is needed in either direction.

- $\lfloor \log_2 m \rfloor + 127$ is already stored in the exponent bits of $m$. Reinterpreting the float32 bits as int32 and shifting right by 23 extracts that field directly, and subtracting 2 gives the E8M0 scale byte.
- $2^{\text{byte} - 127}$ is the float32 whose exponent field equals `byte` and whose mantissa is zero. Shifting `byte` left by 23 and reinterpreting as float32 constructs $X$ directly.

Both bit operations are exact. Floating-point alternatives are not. In float32, `log2` of the largest float below $2^{20}$ is $20 - 8.6 \cdot 10^{-8}$, which rounds up to $20.0$ and makes `floor(log2(m))` too large by one. Similarly, `pow(2.0, e)` is evaluated through logarithms on some backends and can miss an exact power of two by one ulp (observed in `mx.power(2.0, e)` in MLX 0.32.4 for negative integers and in `jnp.exp2`).

**Piecewise rounding.** Because $X$ is an exact power of two, $x / X$ is exact, so the only rounding happens when snapping $a = \min(|x| / X, 6)$ to the FP4 grid. The grid is uniform within each of three regions, so rounding to the nearest grid point is just integer rounding in units of the local step plus a region offset:

$$
2E + M = \begin{cases}
\operatorname{round}(2a) & a < 2, \\
\operatorname{round}(a) + 2 & 2 \le a < 4, \\
\operatorname{round}(a / 2) + 4 & 4 \le a \le 6 .
\end{cases}
$$

Because every offset is even, the parity of the rounded integer matches the mantissa bit $M$. The framework's built-in round-half-to-even therefore implements IEEE 754 `roundTiesToEven` on FP4 directly, as required by section 5.3.3 of the specification.

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

The JAX and MLX implementations are line-for-line translations because all three frameworks support bit reinterpretation via `view`. Where `frexp` and `ldexp` are available (PyTorch and JAX), they provide an equally exact alternative to the bit shifts.

The decoder's `MAGNITUDES` table evaluates the FP4 formula for the eight values of $2E + M$. Its first two entries ($0.0$ and $0.5$) correspond to the subnormal row $E = 0$, where the implicit leading 1 is absent and the exponent matches $E = 1$.

## Common mistakes

- **Computing an exponent per element.** Normalizing each weight by its own power of two encodes every value as if it were the block maximum, destroying all relative magnitudes inside the block.
- **Reducing over the wrong axis or extent.** Taking the maximum over an entire row instead of each 32-element block forces every block in the row to share the largest block's scale.
- **Forgetting to subtract $e_{\max} = 2$.** Using $e = \lfloor \log_2 m \rfloor$ places $m / X$ in $[1, 2)$, leaving the codes for $2$, $3$, $4$, and $6$ unused and turning the format into 3-bit quantization.
- **Choosing a scale that avoids saturation.** Picking the smallest power of two with $m / X \le 6$ raises the scale byte by 1 whenever the maximum would saturate. MLX 0.32.4 (`mx.quantize(w, mode="mxfp4")`) uses this strategy, which section 6.3 permits as an alternative, but it does not match the algorithm specified here.
- **Using `floor(log2(m))` in float32.** This works for most inputs but overestimates the exponent by 1 when $m$ is one ulp below a large power of two.
- **Truncating instead of rounding.** Rounding toward zero doubles the worst-case quantization error and biases weights toward zero.
- **Rounding ties away from zero.** Using `floor(v + 0.5)` rounds $2.5$ to $3$ and $5$ to $6$, whereas round-half-to-even rounds them to $2$ and $4$ (the codes with mantissa bit 0).
- **Omitting saturation.** Without clamping $a$ to $6$, a value of $7$ or more rounds to magnitude code 8, which overflows into the sign bit and decodes the block's largest weight as $-0$.
- **Ignoring subnormals in the encoder.** Treating the grid as starting at $1$ rounds every value below $1$ to $0$ or $1$ and never emits code 1 ($0.5$).
- **Ignoring subnormals in the decoder.** Applying the normal formula $2^{E-1}(1 + M/2)$ to $E = 0$ maps codes 0 and 1 to $0.5$ and $0.75$, turning every zero weight into half a step.
- **Dropping the sign of zero.** IEEE 754 rounding preserves the sign of negative values that round to zero, producing code 8 (`-0`). A decoder outputs the same numeric zero, but byte-exact encoder tests catch the difference.
- **Subtracting 127 in uint8.** Evaluating `exps - 127` before widening wraps around for every byte below 127, which includes every block with scale less than 1. Widen `exps` to int32 first.

## Complexity

Encoding takes one pass to compute the block maxima and one pass to round the elements. Decoding takes one table lookup and one multiplication per element. Storage is $4 + 8 / 32 = 4.25$ bits per weight.

## In the wild

MXFP4 is defined in the *OCP Microscaling Formats (MX) Specification*, version 1.0 (Open Compute Project, September 2023). Section 5.2 specifies the concrete formats (MXFP4 uses FP4 elements, block size 32, and an E8M0 scale), section 5.3.3 defines FP4 encoding, rounding, and saturation, section 5.4.1 defines the E8M0 scale (with byte 255 reserved for NaN), and section 6.3 specifies scalar-to-MX conversion. The specification leaves memory layout to the implementation.

In MLX, `mx.quantize(w, mode="mxfp4")` packs eight 4-bit codes into each uint32 (lowest index in the lowest bits, matching `bit-packing`) and stores scales as uint8, while `mx.dequantize(..., mode="mxfp4")` matches this problem's decoder. The llm-compressor library also ships `MXFP4` among its built-in quantization presets.
