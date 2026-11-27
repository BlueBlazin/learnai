## The idea

A 32-bit word can be viewed as an eight-digit base-16 number where each 4-bit code is a single digit. Putting the lowest index in the lowest bits gives code $i$ in a group a weight of $16^i$, which corresponds to a left shift of $4i$ bits:

$$
\text{word} = \sum_{i=0}^{7} c_i \ll 4i,
\qquad
c_i = (\text{word} \gg 4i) \mathbin{\&} \texttt{0xF} .
$$

Both operations vectorize cleanly by broadcasting against the shift vector `[0, 4, 8, 12, 16, 20, 24, 28]`. To pack, pad the codes with zeros to a multiple of 8, reshape them to `(m, 8)` so each column aligns with its shift amount, shift left, and sum across the columns. Because the shifted nibbles occupy disjoint bit ranges, addition cannot carry and gives the same result as a bitwise OR. To unpack, add a trailing axis to `words`, broadcast the right shifts, mask with `0xF`, and flatten in row-major order. All padding lands at the end of the flattened vector, where slicing to `[:n]` removes it.

Passing `n` alongside `words` is essential because the packed words alone cannot distinguish a genuine trailing zero code from zero padding.

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

The expression `-n % 8` gives the exact number of padding elements needed to reach the next multiple of 8 (and evaluates to 0 when `n` is already a multiple of 8). Because PyTorch's `uint32` type does not support bit shifts, the arithmetic runs in `int64`, which represents every unsigned 32-bit word as a non-negative integer. Converting `uint32` to `int64` zero-extends the value. The JAX and MLX implementations follow the same structure directly in `uint32` using `jnp.pad` and `mx.pad`.

## Common mistakes

- **Putting the high nibble first.** Placing the first code in bits 28 to 31 (the way hex digits are written left to right) yields a pair of functions that round-trip cleanly with each other but read standard checkpoints as garbage. Round-trip tests alone cannot catch layout mismatches, which is why the test suite checks the packed word values directly.
- **Right byte order, wrong nibble order.** Packing each byte high nibble first (`byte = (c0 << 4) | c1`) while keeping bytes in little-endian order is a common slip because it matches how single bytes are written in hex.
- **Shifting before widening.** Input codes have dtype `uint8`. In PyTorch, NumPy, JAX, and MLX, shifting a `uint8` tensor left by 12 stays in `uint8` and discards the bits. Unlike C, tensor libraries do not promote small unsigned integers to `int` before shifting.
- **Sign extension on right shifts.** In a signed 32-bit integer, any word whose position-7 code is 8 or greater is negative. An arithmetic right shift fills the top bits with ones, turning `word >> 28` into a value between $-8$ and $-1$ instead of 8 to 15. Applying `& 0xF` to every position (including the last) masks away those sign bits, and using an unsigned type or `int64` avoids the issue altogether.
- **Returning the padding.** Omitting the `[:n]` slice leaves up to seven extra zeros at the end, causing shape mismatches or injecting phantom weights downstream.
- **Dropping the final partial word.** Truncating `codes` to a multiple of 8 instead of padding silently loses up to seven weights.

## Complexity

Both functions run in $O(n)$ time. The packed output occupies $4 \lceil n / 8 \rceil$ bytes (half a byte per code), cutting memory usage in half compared to `uint8`.

## In the wild

This is the exact 4-bit layout used by MLX. The `mx.quantize` documentation specifies that 4-bit quantization packs 8 elements into an unsigned 32-bit integer with the first element in the 4 least significant bits, the second in bits 4 through 7, and so on. The second public test is a row packed directly by `mx.quantize`. MLX packs each weight matrix row independently and requires row lengths to be multiples of the group size, so real weight matrices need no padding. Its other bit widths follow the same rule, placing code $i$ at bit offset $b \cdot i$ in a little-endian bitstream split into 32-bit words. For widths that do not divide 32 (3, 5, and 6 bits), a single code can straddle two adjacent words.

GGUF's `Q4_0` format also uses little-endian nibbles within each byte, but arranges elements across a 32-weight block differently. Byte $j$ stores weight $j$ in its low nibble and weight $j + 16$ in its high nibble, letting SIMD kernels unpack both halves of the block with a single mask and shift. In every quantized format, the packing layout is designed around the unpacking kernel, so you must read the specification and verify against real packed words.
