## The idea

bfloat16 was designed so that conversion is trivial: it is float32 with the last 16
mantissa bits cut off. Sign, exponent, and bias are identical, so there are no fields
to re-encode, unlike IEEE half precision, which has a 5-bit exponent with bias 15.

**Decoding** appends 16 zero bits. Every bfloat16 value is therefore a float32 value,
and the conversion is exact for every pattern, including subnormals and NaN payloads.

**Encoding** removes 16 bits, and the question is which of the two neighboring
bfloat16 values to return. Because float patterns of one sign are ordered like the
magnitudes they represent, the two neighbors of the pattern $u$ are simply $h = u \gg
16$ and $h + 1$, and the dropped bits $r$ say how far $u$ is from $h$ in units of
$2^{-16}$ of a step. Round-to-nearest-even is then a comparison of $r$ with one half,
`0x8000`, and the usual integer trick does it in one addition:

$$
\text{rounded} = \big(u + \texttt{0x7FFF} + \text{lsb}\big) \gg 16,
\qquad \text{lsb} = (u \gg 16) \mathbin{\&} 1 .
$$

- If $r <$ `0x8000`, then $r + \texttt{0x7FFF} + \text{lsb} < 2^{16}$: no carry, the
  result is $h$.
- If $r >$ `0x8000`, then $r + \texttt{0x7FFF} \ge 2^{16}$: a carry, the result is
  $h + 1$.
- If $r =$ `0x8000`, the sum is $\texttt{0xFFFF} + \text{lsb}$, which carries exactly
  when $h$ is odd, and $h + 1$ is then even.

The carry is the elegant part. When the 7-bit mantissa is all ones and rounds up, the
addition overflows it to zero and increments the exponent: `0x3FFF` (just below 2)
becomes `0x4000` (2.0). At the top, `0x7F7F` becomes `0x7F80`, infinity. And the
largest subnormal becomes the smallest normal number. In every case that is the next
representable value, because that is how IEEE formats are laid out. No field ever has
to be handled separately.

NaN is the exception, because it is not an ordered value. Its payload can sit entirely
in the dropped bits (`0x7F800001` would truncate to `0x7F80`, infinity), and an
all-ones payload would carry through the exponent and the sign (`0x7FFFFFFF` would
round to `0x8000`, which is $-0.0$). So NaN is detected first and replaced by a quiet
NaN.

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

PyTorch has no unsigned 32-bit arithmetic, so the pattern is held in int64, masked to
its low 32 bits: additions cannot overflow and `>>` shifts in zeros. In int32 the same
code needs care, because `>>` is an arithmetic shift there and copies the sign bit. The
JAX and MLX versions use `uint32` directly (`x.view(jnp.uint32)`, `x.view(mx.uint32)`),
where the addition can wrap only for NaN patterns, which `where` replaces anyway.

`view` reinterprets the bytes; `to` and `astype` convert the value. `bits.to(float32)`
would turn the pattern `0x3F80` into the number 16256.0.

## What rounding buys

Truncation is rounding toward zero. Its error is anywhere from 0 to a full bfloat16
step and always has the same sign, so a truncated tensor is systematically smaller in
magnitude than the original, by half a step on average. Round-to-nearest halves the
worst case to half a step and removes the bias. Sending ties to the even neighbor
removes the last bit of it: ties rounded always up would drift upward.

With 7 mantissa bits plus the implicit leading one, a step is $2^{-7}$ of the leading
power of two, so the relative rounding error is at most $2^{-8} \approx 0.4\%$. That is
coarse, about 2.4 decimal digits, and it is why bfloat16 suits weights and activations
but accumulations (sums, norms, logits before a softmax) are still done in float32.

## Common mistakes

- **Truncating** (`u >> 16`). It passes every exactly representable value and is
  wrong, by one in the last place, for about half of all others.
- **Rounding half up** (`(u + 0x8000) >> 16`). It differs from the specification only
  on exact ties, where it always picks the larger magnitude.
- **Taking the fields apart.** Rounding the 7-bit mantissa by itself and reassembling
  sign, exponent, and mantissa loses the carry: `0x3FFFFFFF`, just below 2, comes out
  as `0x3F80`, which is 1.0. Add to the whole pattern.
- **Forgetting NaN.** Without the special case, some NaNs become infinity, some become
  zero, and some stay NaN, depending on the payload.
- **Dropping the sign of a NaN.** This problem keeps it (see below for who else does).
- **Decoding as IEEE half precision.** Reading the two bytes as `float16` gives
  numbers, not errors: `0x3F80` is 1.0 in bfloat16 and 1.875 in half precision.
- **Shifting before widening.** A uint16 shifted left by 16 is 0 (JAX, MLX) or an
  error (PyTorch). In C the operand is promoted to `int` first, which hides this.
- **Converting where you mean to reinterpret**, in either direction.

## Complexity

$O(n)$ time and memory: a handful of elementwise integer operations.

## In the wild

The rounding expression is the one production libraries use. PyTorch's
`round_to_nearest_even` (in `torch/headeronly/util/BFloat16.h`) computes
`rounding_bias = ((U32 >> 16) & 1) + 0x7FFF` and returns `(U32 + rounding_bias) >> 16`;
Eigen's `float_to_bfloat16_rtne`, which TensorFlow and `ml_dtypes` (the NumPy
extension type that JAX uses for `bfloat16`) inherit, is the same two lines. Decoding is
`uint32(bits) << 16` followed by a bit cast in both.

They disagree about NaN, which is why this problem states a rule. Eigen returns
`0xFFC0` or `0x7FC0` according to the sign bit, the rule used here. PyTorch returns
`0x7FC0` for every NaN. MLX 0.32.4 returns `0x7FFF` or `0xFFFF`. All three are NaN, so
no arithmetic can tell them apart; a bit-for-bit comparison of two converted
checkpoints can.

In files, safetensors stores bfloat16 tensors under the dtype name `BF16` as two bytes
per element, little-endian, in row-major order, so reading a tensor is
`np.frombuffer(buffer, dtype="<u2")` followed by the shift you just wrote. NumPy has no
native bfloat16, which is why loaders without the `ml_dtypes` extension do exactly
this. On the hardware side, AVX-512 BF16 and Arm's BF16 extension provide dot products
on bfloat16 inputs that accumulate in float32.
