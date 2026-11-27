## The idea

bfloat16 is designed for fast conversion to and from float32 by dropping the low 16 mantissa bits. Because sign, exponent width, and exponent bias match float32, no fields need to be re-encoded. By contrast, IEEE float16 uses a 5-bit exponent with bias 15 and requires shifting and re-biasing every exponent.

**Decoding** appends 16 zero bits. Every bfloat16 value is already an exact float32 value, including subnormals and NaN payloads.

**Encoding** drops 16 bits and chooses between the two neighboring bfloat16 patterns. Within a fixed sign, IEEE floating-point bit patterns increase monotonically with magnitude. The two bfloat16 neighbors of a 32-bit pattern $u$ are therefore $h = u \gg 16$ and $h + 1$, and the dropped bits $r = u \mathbin{\&} \texttt{0xFFFF}$ measure the fractional position between them in units of $2^{-16}$. Comparing $r$ against half a step (`0x8000`) with round-to-nearest-even reduces to a single integer addition before the shift:

$$
\text{rounded} = \big(u + \texttt{0x7FFF} + \text{lsb}\big) \gg 16,
\qquad \text{lsb} = (u \gg 16) \mathbin{\&} 1 .
$$

- When $r < \texttt{0x8000}$, $r + \texttt{0x7FFF} + \text{lsb} < 2^{16}$, so there is no carry and the result is $h$.
- When $r > \texttt{0x8000}$, $r + \texttt{0x7FFF} \ge 2^{16}$, so the sum carries and the result is $h + 1$.
- When $r = \texttt{0x8000}$, the sum is $\texttt{0xFFFF} + \text{lsb}$, which carries if and only if $h$ is odd, making $h + 1$ even.

Adding directly to the full 32-bit pattern handles carries automatically. When the 7-bit mantissa is all ones and rounds up, the carry zeros the mantissa and increments the exponent. For example, `0x3FFF` (just below 2) becomes `0x4000` (2.0), `0x7F7F` becomes `0x7F80` (infinity), and the largest subnormal becomes the smallest normal number. Because IEEE layouts place representable magnitudes in contiguous integer order, no field needs separate logic.

NaN is the only exception because it is not an ordered value. A NaN whose payload sits in the dropped bits (`0x7F800001`) would truncate to `0x7F80` (infinity), and an all-ones payload (`0x7FFFFFFF`) would carry through the exponent and sign into `0x8000` ($-0.0$). Detecting NaN first and replacing it with a quiet NaN avoids both failures.

## Reference solution (PyTorch)

```python
def bf16_to_f32(bits):
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x):
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
```

PyTorch lacks unsigned 32-bit arithmetic, so the solution holds the bit pattern in `int64` masked to its low 32 bits. Additions cannot overflow and `>>` shifts in zeros. Doing the same in `int32` requires care because `>>` is an arithmetic right shift that copies the sign bit. JAX and MLX can use `uint32` directly (`x.view(jnp.uint32)`, `x.view(mx.uint32)`), where the addition can wrap only for NaN patterns that `where` replaces anyway.

Note the distinction between `view` and `to`/`astype`. `view` reinterprets the raw bytes without changing bits, whereas `to` and `astype` convert numeric values (`bits.to(float32)` would turn `0x3F80` into `16256.0`).

## What rounding buys

Truncation rounds toward zero. Its error ranges from 0 to a full bfloat16 step and always points toward zero, shrinking magnitudes by half a step on average. Round-to-nearest cuts the worst-case error to half a step and removes that bias, and rounding ties to even removes the remaining upward drift on exact half-steps.

With 7 mantissa bits plus the implicit leading 1, one step is $2^{-7}$ of the current power of two, giving a maximum relative rounding error of $2^{-8} \approx 0.4\%$ (about 2.4 decimal digits). That precision is enough for storing weights and activations, while accumulations such as sums, norms, and pre-softmax logits stay in float32.

## Common mistakes

- **Truncating (`u >> 16`).** Passes every exactly representable value but is off by one in the last place for roughly half of all other inputs.
- **Rounding half up (`(u + 0x8000) >> 16`).** Disagrees with round-to-nearest-even on exact ties by always picking the larger magnitude.
- **Splitting and reassembling fields.** Rounding the 7-bit mantissa in isolation drops the carry into the exponent, turning `0x3FFFFFFF` (just below 2) into `0x3F80` (1.0). Add to the full bit pattern instead.
- **Forgetting NaN.** Depending on the payload, unguarded rounding turns some NaNs into infinity, some into zero, and leaves others as NaN.
- **Dropping the sign of a NaN.** This specification preserves the input sign bit on NaN.
- **Decoding as IEEE float16.** Interpreting bfloat16 bytes as `float16` silently produces wrong values (`0x3F80` is `1.0` in bfloat16 and `1.875` in float16).
- **Shifting before widening.** Left-shifting a `uint16` by 16 yields `0` in JAX and MLX and raises an error in PyTorch. C hides this by promoting operands to `int` first.
- **Converting instead of reinterpreting.** Using `to` or `astype` where a bit-level `view` is required breaks either direction.

## Complexity

$O(n)$ time and memory for $n$ elements, consisting of a small fixed number of elementwise integer operations.

## In the wild

Production libraries use this exact rounding formula. PyTorch's `round_to_nearest_even` (in `torch/headeronly/util/BFloat16.h`) computes `rounding_bias = ((U32 >> 16) & 1) + 0x7FFF` and returns `(U32 + rounding_bias) >> 16`. Eigen's `float_to_bfloat16_rtne`, which TensorFlow and `ml_dtypes` (the NumPy extension type backing JAX's `bfloat16`) inherit, uses the same two lines. Both decode with `uint32(bits) << 16` followed by a bit cast.

Libraries differ on NaN encoding, which is why this problem fixes a rule. Eigen returns `0xFFC0` or `0x7FC0` based on the sign bit (the rule used here), PyTorch returns `0x7FC0` for every NaN, and MLX 0.32.4 returns `0x7FFF` or `0xFFFF`. Arithmetic cannot distinguish these NaN patterns, though a bit-for-bit checkpoint comparison can.

In checkpoint files, `safetensors` stores bfloat16 tensors under the `BF16` dtype as little-endian two-byte values in row-major order. Loaders without `ml_dtypes` read them via `np.frombuffer(buffer, dtype="<u2")` and apply this exact shift. In hardware, AVX-512 BF16 and Arm's BF16 extension provide instructions that compute dot products on bfloat16 inputs and accumulate into float32.
