Think of one word as an eight-digit hexadecimal number. Each code is one digit. Where
does the first code go if the lowest index must end up in the lowest bits, and by how
many bits must code number $i$ of a group be moved to reach its place?

---

`pack4`: pad the codes with zeros to a multiple of 8, reshape to `(m, 8)`, shift
column $i$ left by $4i$ bits, and combine the eight columns (the shifted codes do not
overlap, so OR and addition give the same word). Before you shift, look at the dtype:
the codes are uint8, and a uint8 shifted left by 12 does not become a 16-bit number.
Convert to a type with at least 32 usable bits first.

`unpack4`: the reverse. Broadcast the words against the eight shifts to get an
`(m, 8)` table of codes, flatten it, and keep the first `n`.

---

With `shifts = [0, 4, 8, 12, 16, 20, 24, 28]`:

$$
\text{words} = \sum_i \big(\text{codes.reshape}(m, 8)_{:, i} \ll \text{shifts}_i\big),
\qquad
\text{codes} = \big(\text{words}_{:, \text{None}} \gg \text{shifts}\big) \mathbin{\&} \texttt{0xF} .
$$

Watch the sign on the way back. If the words are held in a signed 32-bit type, a word
with bit 31 set is negative and `>>` fills with ones from the left. Either keep the
words unsigned (uint32 in JAX and MLX, int64 in PyTorch, where a uint32 converts
without becoming negative), or make sure the mask `& 0xF` is applied to every
position, including the last one.
