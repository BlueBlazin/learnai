Implement two functions that pack a vector of 4-bit quantization codes into unsigned 32-bit words (eight codes per word) and unpack those words back into individual codes. Both functions must follow a fixed little-endian 4-bit layout and use the specified integer dtypes.

## Specification

```python
def pack4(codes): ...  # uint8 codes, shape (n,) -> uint32 words, shape (ceil(n / 8),)
def unpack4(words, n): ...  # uint32 words -> the first n uint8 codes
```

**`pack4(codes)`** takes a 1-D uint8 tensor `codes` containing $n \ge 1$ values $c_0, \dots, c_{n-1}$, each in $[0, 15]$. Let $m = \lceil n / 8 \rceil$ be the number of output words, and pad the sequence with trailing zeros to length $8m$ so that $c_j = 0$ for $n \le j < 8m$. Word $k$ stores the eight codes $c_{8k}, \dots, c_{8k+7}$ with the lowest index in the lowest bits:

$$
\text{words}_k = \sum_{i=0}^{7} c_{8k+i} \cdot 16^{\,i}, \qquad 0 \le k < m .
$$

Equivalently, code $c_{8k+i}$ occupies bits $4i$ through $4i + 3$ of word $k$, where bit 0 is the least significant bit:

| bits of word $k$ | 31 to 28 | 27 to 24 | 23 to 20 | 19 to 16 | 15 to 12 | 11 to 8 | 7 to 4 | 3 to 0 |
|---|---|---|---|---|---|---|---|---|
| code | $c_{8k+7}$ | $c_{8k+6}$ | $c_{8k+5}$ | $c_{8k+4}$ | $c_{8k+3}$ | $c_{8k+2}$ | $c_{8k+1}$ | $c_{8k}$ |

Written in hexadecimal, a word therefore displays its eight codes from right to left. This layout is called little-endian nibble order (each 4-bit group is a nibble). When the word is stored in memory in little-endian byte order, byte $b$ holds code $2b$ in its low four bits and code $2b + 1$ in its high four bits.

`pack4` returns a **uint32** tensor of shape `(m,)`. Any unused padding positions in the last word must be zero bits.

**`unpack4(words, n)`** is the inverse of `pack4`. It takes a uint32 tensor `words` of shape `(m,)` and a Python `int` `n` with $1 \le n \le 8m$, and returns a **uint8** tensor of shape `(n,)` whose entries are:

$$
c_j = \left\lfloor \frac{\text{words}_{\lfloor j / 8 \rfloor}}{16^{\,j \bmod 8}} \right\rfloor \bmod 16,
\qquad 0 \le j < n .
$$

Only the first `n` codes are returned, and any remaining bits in `words` beyond position $n - 1$ are ignored. For every valid `codes`, `unpack4(pack4(codes), len(codes))` must equal `codes`.

Each word is an unsigned 32-bit integer, so a code of 8 or more in position 7 sets bit 31.

> [!NOTE]
> PyTorch can store and convert uint32 tensors but implements no bit-shift operations for them. Perform the shifts in a wider integer type and convert to uint32 at the end.

## Shapes

| name | shape | dtype |
|---|---|---|
| `codes` | `(n,)`, $n \ge 1$ | uint8, values in $[0, 15]$ |
| result of `pack4` | `(ceil(n / 8),)` | uint32 |
| `words` | `(m,)`, $m \ge 1$ | uint32 |
| `n` | Python `int`, $1 \le n \le 8m$ | |
| result of `unpack4` | `(n,)` | uint8, values in $[0, 15]$ |

## Example

```python
codes = [3, 14, 0, 9, 5, 12, 1, 6, 15, 2, 8]  # n = 11

pack4(codes)  # [0x61C590E3, 0x0000082F]
unpack4([0x61C590E3, 0x0000082F], 11)  # [3, 14, 0, 9, 5, 12, 1, 6, 15, 2, 8]
```

Eleven codes require $\lceil 11 / 8 \rceil = 2$ words. The first word holds codes 0 through 7 (`3, 14, 0, 9, 5, 12, 1, 6`), which correspond to the hexadecimal digits `3, E, 0, 9, 5, C, 1, 6`. Placing the first code in the lowest bits produces `0x61C590E3`:

$$
3 + 14 \cdot 16 + 0 \cdot 16^2 + 9 \cdot 16^3 + 5 \cdot 16^4 + 12 \cdot 16^5 + 1 \cdot 16^6 + 6 \cdot 16^7 = 1{,}640{,}337{,}635 .
$$

The second word holds the remaining three codes (`15, 2, 8`) followed by five zero-padded positions, giving `0x0000082F`. Calling `unpack4` with `n = 11` returns the eleven original codes and drops the five padding zeros.

![The two words of the example. Code $c_0$ sits in the lowest four bits of word 0, and the five unused fields of word 1 are zero.](assets/word-layout.svg)

## Constraints

- All comparisons are **exact**.
- $1 \le n \le 100{,}000$. Every code is in $[0, 15]$.
- `pack4` must return dtype uint32 and `unpack4` must return dtype uint8.
- No Python loop over the codes or the words is needed (a loop over the eight positions of a word is fine).

## Why this matters

A 4-bit model only achieves its 2x memory savings over 8-bit storage when the codes are actually packed into four bits, because standard tensor libraries do not provide a native 4-bit dtype. MLX stores 4-bit quantized weights as uint32 words in this exact layout, and its Metal and CPU kernels unpack them on the fly during matrix multiplication.

The bit layout is a strict contract between the code that writes a checkpoint and the kernel that reads it. If the nibble order is wrong, the model still loads and every weight unpacks to a valid code, yet the outputs become noise. If signedness is handled incorrectly, only weights in the top nibble of a word with a value of 8 or greater (roughly one in sixteen weights) get corrupted, which degrades quality quietly and is notoriously hard to debug.
