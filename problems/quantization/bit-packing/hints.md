Think of each 32-bit word as an eight-digit hexadecimal number where each 4-bit code is one digit. If the lowest index must land in the lowest bits, where does the first code go, and how many bits left must code $i$ of a group move to reach its slot?

---

For `pack4`, pad `codes` with trailing zeros to a multiple of 8, reshape to `(m, 8)`, shift column $i$ left by $4i$ bits, and combine the eight columns. Because the shifted codes occupy disjoint bit positions, bitwise OR and addition produce the same word. Before shifting, cast `codes` from `uint8` to a type with at least 32 usable bits, because shifting a `uint8` left by 12 discards its bits instead of widening the dtype.

For `unpack4`, reverse the steps. Broadcast `words` against the eight shift amounts to produce an `(m, 8)` array of codes, flatten it, and slice to the first `n` elements.

---

With `shifts = [0, 4, 8, 12, 16, 20, 24, 28]`:

$$
\text{words} = \sum_i \big(\text{codes.reshape}(m, 8)_{:, i} \ll \text{shifts}_i\big),
\qquad
\text{codes} = \big(\text{words}_{:, \text{None}} \gg \text{shifts}\big) \mathbin{\&} \texttt{0xF} .
$$

Watch out for signedness when unpacking. In a signed 32-bit type, a word with bit 31 set is negative, so `>>` sign-extends with ones from the left. Keep the words unsigned (`uint32` in JAX and MLX, or `int64` in PyTorch where `uint32` converts without becoming negative), and make sure `& 0xF` is applied to every position, including the top nibble.
