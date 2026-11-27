Write `1.0` as a float32 bit pattern (`0x3F800000`) and as a bfloat16 bit pattern (`0x3F80`). The two formats share the sign bit and the entire 8-bit exponent, differing only in how many mantissa bits follow. Think about which single integer operation turns the 16-bit pattern into the 32-bit pattern, and what must happen to the dtype before you apply it.

---

For `bf16_to_f32`, widen the `uint16` tensor to a 32-bit (or wider) integer type, shift left by 16, and reinterpret the 32 bits as `float32`. Use `view` to reinterpret raw bits rather than `astype` or `to`, which convert numeric values.

For `f32_to_bf16`, reinterpret `x` as a 32-bit integer and split it into the kept bits `h = u >> 16` and the dropped bits `r = u & 0xFFFF`. Compare `r` against `0x8000`, which is half of one unit of `h`. If you store `u` in a signed 32-bit integer, remember that `>>` copies the sign bit into the top bits, so mask the result or widen to `int64` first.

---

The entire rounding rule fits in one addition before the shift:

$$
\text{rounded} = \big(u + \texttt{0x7FFF} + ((u \gg 16) \mathbin{\&} 1)\big) \gg 16 .
$$

Adding `0x7FFF` carries into the kept bits whenever $r > \texttt{0x8000}$. The extra `1`, present only when the kept part is odd, also makes it carry at $r = \texttt{0x8000}$. Keep the bit pattern intact rather than splitting fields, so a carry out of the mantissa automatically increments the exponent. Make sure the sum cannot overflow the integer type holding `u`.

Detect NaN separately from the bit pattern using `(u & 0x7FFFFFFF) > 0x7F800000`. For NaN elements, return `0x7FC0`, plus `0x8000` if bit 31 of `u` is set.
