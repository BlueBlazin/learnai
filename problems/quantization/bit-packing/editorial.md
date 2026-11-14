## The idea

A word is a number in base 16 with eight digits, and each 4-bit code is one digit.
"Lowest index in the lowest bits" means that code $i$ of a group is the digit with
weight $16^i$, so it is moved to its place by a left shift of $4i$ bits:

$$
\text{word} = \sum_{i=0}^{7} c_i \ll 4i,
\qquad
c_i = (\text{word} \gg 4i) \mathbin{\&} \texttt{0xF} .
$$

Both directions vectorize as one broadcast against the vector of shifts
`[0, 4, ..., 28]`. For packing, reshape the padded codes to `(m, 8)` so that the shift
vector lines up with the positions inside a group, and reduce over that axis. The
shifted codes occupy disjoint bits, so a sum is the same as a bitwise OR and cannot
carry. For unpacking, add an axis to the words, shift, mask, and flatten: row-major
order puts the codes back in sequence, with the padding at the very end, where slicing
to `n` removes it.

Padding with zeros and passing `n` separately is what makes the format self-consistent:
the words alone cannot say whether a trailing zero is a code or a filler.

## Reference solution (PyTorch)

```python
SHIFTS = 4 * torch.arange(8)


def pack4(codes):
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words, n):
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    return nibbles.reshape(-1)[:n].to(torch.uint8)
```

`-n % 8` is the number of codes missing to the next multiple of 8 (0 when `n` already
is one). PyTorch's uint32 is a storage type without shifts, so the arithmetic is done
in int64, which holds every 32-bit word as a non-negative number; the conversion
`uint32 -> int64` zero-extends. The JAX and MLX versions do the same in `uint32`
throughout, with `jnp.pad` / `mx.pad`.

## Common mistakes

- **High nibble first.** Putting the first code in bits 28 to 31, the way one writes
  digits, gives a consistent pair of functions that round-trips perfectly and reads
  everyone else's files as garbage. A round-trip test alone can never catch a layout
  bug, which is why the tests also compare the words themselves.
- **The right bytes, the wrong nibbles.** Filling each byte high nibble first
  (`byte = c0 << 4 | c1`) while keeping the bytes in order. It looks natural because
  it is how a byte is written in hex.
- **Shifting before widening.** The codes are uint8. In PyTorch `codes << 12` with a
  Python integer stays uint8, and the bits are gone. NumPy, JAX, and MLX behave the
  same way. In C the operand would be promoted to `int` first, which is why code
  ported from C has this bug.
- **Sign extension.** In a signed 32-bit type, a word whose position-7 code is 8 or
  more is negative, and `>>` is an arithmetic shift that fills with copies of the sign
  bit. `word >> 28` is then a number from $-8$ to $-1$, not from 8 to 15. A mask on
  every position repairs it; assuming that "nothing lies above the top nibble" does
  not. Unsigned types avoid the question.
- **Returning the padding.** Without the slice to `n`, the result has up to seven
  extra zeros, which downstream becomes a shape error at best and seven phantom
  weights at worst.
- **Dropping the last partial word**, by truncating to a multiple of 8 instead of
  padding.

## Complexity

$O(n)$ time. The packed vector takes $4 \lceil n / 8 \rceil$ bytes, half a byte per
code, against one byte per code before.

## In the wild

This is the layout of MLX's quantized weights. The documentation of `mx.quantize`
states that for 4-bit quantization "we fit 8 elements in an unsigned 32 bit integer
where the 1st element occupies the 4 least significant bits, the 2nd bits 4-7 etc.",
and the second public test is a row packed by `mx.quantize` itself. MLX packs each row
of a weight matrix separately and requires the row length to be a multiple of the
group size, so real rows never need padding. Its other bit widths follow the same
rule, code $i$ of a row at bit offset $b \cdot i$ of one long little-endian bit string
that is then cut into 32-bit words. For widths that do not divide 32 (3, 5, and 6
bits) a code can therefore straddle two words, which the 4-bit case never has to
handle.

GGUF's `Q4_0` blocks also store 4-bit codes low nibble first, but arrange them
differently: within a block of 32 weights, byte $j$ holds weight $j$ in its low nibble
and weight $j + 16$ in its high nibble, so that a SIMD kernel can split a block into
its two halves with one mask and one shift. The lesson of both formats is the same:
the layout is chosen for the unpacking kernel, and the only way to know it is to read
the specification, then test against bytes that the real writer produced.
