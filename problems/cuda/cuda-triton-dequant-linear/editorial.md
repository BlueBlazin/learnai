## The idea

A dequantized weight is needed for exactly one multiplication. So compute it where
it is used, in the registers of the program that multiplies, and let it go: the
kernel's memory traffic is the packed words, the scales and biases, and the
activations, and its only store is the result.

**One program per output element.** $y_{b,o}$ is a dot product of row $b$ of `x`
with row $o$ of $\hat W$, independent of every other element, so the grid is
`(n_out, batch)` and each program runs the block loop of `cuda-triton-rmsnorm` over
the `n_in` columns with a float32 accumulator and a masked tail.

**Three indices per column.** For column $c$ of output row $o$:

| what | where | why |
|---|---|---|
| the word | `words_row + c // 8` | eight codes per uint32 |
| the code in it | `(word >> 4 * (c % 8)) & 0xF` | code 0 is in the lowest bits |
| scale and bias | `scales_row + c // group_size` | one pair per group of columns |

All three are computed for a whole block at once: `cols // 8`, `cols % 8`, and
`cols // group_size` are blocks of integers, and a load through a block of offsets
reads one element per entry, whether or not entries repeat. That is the important
step up from the previous problem, where every load was a contiguous run: here the
offsets are a function of the column, and nothing ties them to the block. A block
of 16 columns under groups of 24 changes group in mid-block; a block of 64 under
groups of 8 spans eight groups. The kernel never needs to know.

**Unsigned shifts.** The word is loaded as `tl.uint32` and stays unsigned, so `>>`
brings in zeros. The shift amount is converted to `tl.uint32` as well; mixing a
signed and an unsigned integer of the same width promotes to unsigned in Triton, so
this is for the reader more than for the compiler. A mask after the shift (`& 0xF`)
is robust to either kind of shift; the two-shift idiom of signed bit-field code,
`(word << a) >> b`, is not, and the shim refuses it as soon as the shifted value is
negative.

**float32 arithmetic.** The code becomes a float32, the scale and bias are converted
on load, and the weight, the product, and the partial sums are float32 whatever the
inputs are. A float16 accumulator loses on both ends: each addition rounds the
partial sum to 11 bits, and a sum of a few hundred products of order 100 passes
65504.

**Masked columns contribute zero** because `x` is loaded with `other=0.0`; the other
three loads get an `other=` too, so that no undefined value is ever multiplied, even
by zero.

## Reference solution (PyTorch)

```python
@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
):
    out_row = tl.program_id(0)
    batch_row = tl.program_id(1)
    x_row = x_ptr + batch_row * n_in
    words_row = words_ptr + out_row * (n_in // 8)
    scales_row = scales_ptr + out_row * (n_in // group_size)
    biases_row = biases_ptr + out_row * (n_in // group_size)

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_in, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_in
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        word = tl.load(words_row + cols // 8, mask=mask, other=0)
        code = (word >> ((cols % 8) * 4).to(tl.uint32)) & 0xF
        group = cols // group_size
        scale = tl.load(scales_row + group, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(biases_row + group, mask=mask, other=0.0).to(tl.float32)
        acc += x * (code.to(tl.float32) * scale + bias)
    tl.store(y_ptr + batch_row * n_out + out_row, tl.sum(acc, axis=0))


def qlinear(x, words, scales, biases, group_size, block=64):
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    rows = x.reshape(-1, n_in).contiguous()
    y = torch.empty((rows.shape[0], n_out), dtype=x.dtype, device=x.device)
    grid = (n_out, rows.shape[0])
    qlinear_kernel[grid](
        rows,
        words.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        y,
        n_in,
        n_out,
        group_size,
        BLOCK=block,
    )
    return y.reshape(*x.shape[:-1], n_out)
```

This is the straightforward kernel, and it is the one to get right first. It loads
each word eight times and each scale `group_size` times; those loads hit the same
cache line, but they are not free. Two refinements keep the structure and cut the
traffic, at the price of tying `BLOCK` to the format:

- load `BLOCK / 8` words once and expand them against the eight shifts as a
  two-dimensional block (`word[:, None] >> shifts[None, :]`);
- use the identity of `quantized-matmul`,
  $\sum_{c \in g} x_c (q_c s_g + b_g) = s_g \sum x_c q_c + b_g \sum x_c$, with one
  block per group, so that the scale and bias are scalars loaded once per group and
  the inner product is between activations and raw codes.

## Common mistakes

- **Codes read from the high bits down** (`4 * (7 - c % 8)`). Every eight weights
  are reversed.
- **One scale and bias per block**, taken from the block's first column. Right
  while every block lies inside one group, so right for `BLOCK = group_size` and
  wrong as soon as someone tunes the block size.
- **Scales indexed without the row offset.** Every output row uses row 0's scales:
  right for a one-row test.
- **A signed shift.** With an arithmetic shift, codes of 8 and above come out as
  $-8$ to $-1$; the shim stops at the shift itself.
- **The bias outside the product** (`x * code * scale + bias`). The bias is part of
  the weight and must be multiplied by the activation; this adds each column's bias
  itself, $\text{group\_size} \cdot b$ per group where $b \sum x$ belongs.
- **A float16 accumulator.** Right to three digits on short rows.
- **No mask**, or **masks without `other=`**. The first reads the next row's
  activations and codes into this output and runs off the tensors at the end; the
  second multiplies undefined values. On a GPU neither fails.
- **`y` stored transposed** (`o * batch + b`), or a launch grid with the axes
  swapped.
- **Dequantizing in the wrapper** and calling `matmul`. The right numbers, and the
  float matrix back in memory.
- **Strided tensors passed to the kernel**, or a float32 output for float16
  activations.

## Complexity

`n_out * batch` programs. Each loads `4 * n_in` elements (an activation, a word, a
scale, and a bias per column) in $\lceil \text{n\_in} / \text{BLOCK} \rceil$
iterations and stores one. Memory beyond the inputs: the output.

## In the wild

MLX's `quantized_matmul` is this operation for the same layout; its Metal
counterpart for dequantization alone is `mlx-metal-kernel-dequant`, which has one
thread per weight where this kernel has one program per output. GPTQ-style formats
pack 4-bit codes into int32 words, which is where the signed-shift bug comes from in
practice, and store a zero point per group instead of a bias; GGUF's `Q4_K` packs
blocks of 32 codes with 6-bit scales. The loop is the same in all of them; the three
index expressions are what changes. A production kernel would also tile the batch
(one program computing several outputs against the same unpacked codes) so that
each word is unpacked once per tile and not once per output element.

What was verified where. On the CPU, under the kit's `tritonshim`: everything the
judge checks. The shim follows Triton's documented type rules (the unsigned type
wins between a signed and an unsigned integer of one width; a Python integer next
to a block takes the block's type; integer `//` and `%` round toward zero), and
assumes, without a documented statement, that `>>` is a logical shift for unsigned
integers (a negative signed value it refuses to shift at all), and that a torch
`uint32` tensor is accepted as a `uint32` pointer. The reference kernel never shifts
a signed value, so it depends only on the first assumption. The GPU cases, which
hand this kernel to the real compiler, have never been run.
