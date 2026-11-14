Pack a vector of 4-bit quantization codes into 32-bit words, eight codes per word, and
unpack them again. The arithmetic is two shifts and a mask; what has to be exactly
right is the layout, because the packed words are a storage format that other code
reads, and the integer types, because the codes arrive in 8 bits and leave in 32.

## Specification

```python
def pack4(codes): ...  # uint8 codes, shape (n,) -> uint32 words, shape (ceil(n / 8),)
def unpack4(words, n): ...  # uint32 words -> the first n uint8 codes
```

**`pack4(codes)`**. `codes` holds $n \ge 1$ values $c_0, \dots, c_{n-1}$, each in
$[0, 15]$. Let $m = \lceil n / 8 \rceil$, and extend the sequence with zeros to length
$8m$: $c_j = 0$ for $n \le j < 8m$. Word $k$ holds the eight codes $c_{8k}, \dots,
c_{8k+7}$, the lowest index in the lowest bits:

$$
\text{words}_k = \sum_{i=0}^{7} c_{8k+i} \cdot 16^{\,i}, \qquad 0 \le k < m .
$$

Equivalently, code $c_{8k+i}$ occupies bits $4i$ to $4i + 3$ of word $k$ (bit 0 is
the least significant bit):

| bits of word $k$ | 31 to 28 | 27 to 24 | 23 to 20 | 19 to 16 | 15 to 12 | 11 to 8 | 7 to 4 | 3 to 0 |
|---|---|---|---|---|---|---|---|---|
| code | $c_{8k+7}$ | $c_{8k+6}$ | $c_{8k+5}$ | $c_{8k+4}$ | $c_{8k+3}$ | $c_{8k+2}$ | $c_{8k+1}$ | $c_{8k}$ |

Written in hexadecimal, a word therefore shows its eight codes from right to left.
This is little-endian nibble order: if the word is stored in memory in little-endian
byte order, byte $b$ of the buffer holds code $2b$ in its low four bits and code
$2b + 1$ in its high four bits.

It returns a **uint32** tensor of shape `(m,)`. The padding positions of the last word
are zero bits.

**`unpack4(words, n)`** is the inverse. `words` is a uint32 tensor of shape `(m,)` and
`n` is a Python integer with $1 \le n \le 8m$. It returns a **uint8** tensor of shape
`(n,)` with

$$
c_j = \left\lfloor \frac{\text{words}_{\lfloor j / 8 \rfloor}}{16^{\,j \bmod 8}} \right\rfloor \bmod 16,
\qquad 0 \le j < n .
$$

Only the first `n` codes are returned. Whatever the remaining bits of `words` hold
(zeros when the words come from `pack4`, anything otherwise) is ignored. For every
valid `codes`, `unpack4(pack4(codes), len(codes))` equals `codes`.

A word is an unsigned 32-bit number: a code of 8 or more in position 7 sets bit 31.

> [!NOTE]
> PyTorch can store and convert uint32 tensors but implements no shifts for them.
> Compute in a wider integer type and convert to uint32 at the end.

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

Eleven codes need $\lceil 11 / 8 \rceil = 2$ words. The first word holds codes 0 to 7.
In hexadecimal, 3, 14, 0, 9, 5, 12, 1, 6 are the digits `3 E 0 9 5 C 1 6`; with the
first code in the lowest position the word reads `0x61C590E3`:

$$
3 + 14 \cdot 16 + 0 \cdot 16^2 + 9 \cdot 16^3 + 5 \cdot 16^4 + 12 \cdot 16^5 + 1 \cdot 16^6 + 6 \cdot 16^7 = 1{,}640{,}337{,}635 .
$$

The second word holds the remaining codes 15, 2, 8 and five padding zeros:
`0x0000082F`. Unpacking with `n = 11` returns the eleven codes and not the padding.

## Constraints

- All comparisons are **exact**.
- $1 \le n \le 100{,}000$. Every code is in $[0, 15]$.
- `pack4` must return dtype uint32 and `unpack4` dtype uint8.
- No Python loop over the codes or the words is needed (a loop over the eight
  positions of a word is fine).

## Why this matters

A 4-bit model is only a quarter of the size of an 8-bit one if the codes are really
stored in four bits, and no tensor library has a 4-bit dtype. So every 4-bit format
packs: MLX stores quantized weights as uint32 words in exactly this layout, and its
Metal and CPU kernels unpack them on every matrix multiplication. The layout is a
contract between the code that writes a checkpoint and the kernel that reads it. Get
the nibble order wrong and the model still loads, every weight is a valid code, and the
output is noise. Get the signedness wrong and only the weights whose top code is 8 or
more are corrupted, about one in sixteen, which is enough to degrade a model and little
enough to be very hard to find.
