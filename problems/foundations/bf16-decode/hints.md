Write 1.0 as a float32 bit pattern (`0x3F800000`) and as a bfloat16 bit pattern
(`0x3F80`). The formats share the sign bit and the whole exponent, and differ only in
how many mantissa bits follow. What single integer operation turns the 16-bit pattern
into the 32-bit one, and what must happen to the dtype before you apply it?

---

`bf16_to_f32`: convert the uint16 tensor to a 32-bit (or wider) integer type, shift
left by 16, and reinterpret the 32 bits as float32. Reinterpreting is not converting:
you want `view`, which keeps the bits, not `astype` or `to`, which keeps the value.

`f32_to_bf16`: reinterpret `x` as a 32-bit integer and split it into the kept part
`h = u >> 16` and the dropped part `r = u & 0xFFFF`. Compare `r` with `0x8000`, half
of one unit of `h`. If you hold `u` in a signed 32-bit type, remember that `>>` copies
the sign bit into the top, so mask the result or use a wider type.

---

The whole rounding rule fits in one addition, done before the shift:

$$
\text{rounded} = \big(u + \texttt{0x7FFF} + ((u \gg 16) \mathbin{\&} 1)\big) \gg 16 .
$$

Adding `0x7FFF` carries into the kept part exactly when $r >$ `0x8000`; the extra 1,
present only when the kept part is odd, makes it carry at $r =$ `0x8000` as well. Do
not take the fields apart: adding to the whole pattern lets a carry out of the
mantissa increment the exponent, which is what must happen. Make sure the sum cannot
overflow the type you hold `u` in.

NaN is selected separately, from the bits: `(u & 0x7FFFFFFF) > 0x7F800000`. For those
elements return `0x7FC0`, plus `0x8000` if bit 31 of `u` is set.
