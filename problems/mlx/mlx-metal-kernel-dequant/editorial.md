## The idea

Dequantization is embarrassingly parallel: every output weight depends on one code,
one scale, and one bias, and on nothing any other weight computes. So the kernel is
one thread per weight, and the design question is only how a thread finds its three
inputs from its position.

Launch a two-dimensional grid, `grid=(n_in, n_out, 1)`. Thread
$(x, y) = (\text{col}, \text{row})$ then needs four integers:

| what | value | why |
|---|---|---|
| word | `row * n_words + col / per_word` | `per_word = 32 / bits` codes per word; rows are stored one after another |
| shift | `bits * (col % per_word)` | code 0 of a word is in the lowest bits |
| group | `row * n_groups + col / group_size` | scales and biases are `(out, groups)`, row-major |
| output | `row * n_in + col` | the output is row-major too |

and one line of arithmetic, `code * scale + bias`.

Three details make it exact and fast.

- **Unsigned everything.** `word >> shift` on a `uint` shifts in zeros. On an `int`
  it copies the sign bit, which goes unnoticed as long as a mask follows, and breaks
  the moment someone isolates the field with two shifts (`(word << a) >> b`), the
  idiom from signed bit-field code: every code with its top bit set comes out
  negative.
- **`bits` as a template int.** `template=[("BITS", bits)]` makes `32 / BITS` and
  the mask compile-time constants, so the divisions and modulos by `per_word` become
  shifts and masks in the compiled kernel. Passing `bits` as a 0-d array input would
  work, and would do real integer divisions in every thread.
- **Compute in `float`, convert once.** The output type `T` is float16, bfloat16, or
  float32. `fma(float(code), float(scale), float(bias))` rounds once in float32 and
  `T(...)` rounds once more to the output type. This is bit for bit what
  `mx.dequantize` returns for all three dtypes on this machine. Computing in `T`
  would not compile for bfloat16 (MLX's Metal type for it is a storage type with
  conversions, not arithmetic) and would round the product of a 255-valued code and
  a scale to 11 bits in half precision before the bias is added.

## Reference solution (MLX)

```python
_SOURCE = """
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint per_word = 32 / BITS;              // codes in one uint32 word
    uint n_words = words_shape[1];          // words in one row
    uint n_groups = scales_shape[1];        // groups in one row
    uint n_in = n_words * per_word;         // weights in one row

    uint word = words[row * n_words + col / per_word];
    uint code = (word >> (BITS * (col % per_word))) & ((1u << BITS) - 1u);

    uint group = row * n_groups + col / (n_in / n_groups);
    float w = fma(float(code), float(scales[group]), float(biases[group]));
    out[row * n_in + col] = T(w);
"""

_KERNEL = mx.fast.metal_kernel(
    name="metal_dequantize",
    input_names=["words", "scales", "biases"],
    output_names=["out"],
    source=_SOURCE,
)


def metal_dequantize(words, scales, biases, group_size, bits):
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    return _KERNEL(
        inputs=[words, scales, biases],
        template=[("T", scales.dtype), ("BITS", bits)],
        grid=(n_in, n_out, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(n_out, n_in)],
        output_dtypes=[scales.dtype],
    )[0]
```

The kernel reads the row widths from `words_shape` and `scales_shape`, which MLX adds
to the signature because the body mentions them, so one compiled kernel per
`(dtype, bits)` serves every layer of a model. The largest flat index is
`out * in`, 16.8 million for a `4096 x 4096` layer and 257 million for the
`100352 x 2560` embedding of a 3B model, both inside a `uint`.

`ensure_row_contiguous` stays at its default: a strided input is copied before the
kernel runs, and the flat indices are right. The packed weights of a checkpoint are
always contiguous, so nothing is copied in the case that matters.

## What the alternatives cost

Measured on this machine (M1 Max, MLX 0.32.4), a `4096 x 4096` layer at 4 bits in
groups of 64 (8 MiB of words, a 64 MiB float32 result), steady state:

| implementation | time per call | peak memory above the inputs |
|---|---|---|
| this kernel, one thread per weight | 0.50 ms | 64 MiB |
| `mx.dequantize` | 0.43 ms | 64 MiB |
| MLX operations: shift, mask, reshape, cast, `repeat`, multiply, add | 2.8 ms | 192 MiB |
| a kernel with one thread per row, looping over columns | 3.7 ms | 64 MiB |
| the row kernel called from a Python loop, 4096 launches | 39 ms | 128 MiB |

The hand-written kernel matches MLX's own. The composition of built-in operations is
five times slower and holds three matrices at its peak, because every step
materializes a full-size temporary: the unpacked codes as uint32, the codes as
floats, the repeated scales. This is where a custom kernel earns its keep, unlike the
elementwise chain of the previous problem.

The threadgroup shape hardly matters here: `(32, 8, 1)`, `(16, 16, 1)`, `(256, 1,
1)`, and `(1024, 1, 1)` all measure 0.50 ms. What matters is that the work is spread
over 16.8 million short threads and not 4096 long ones.

## Common mistakes

- **Codes read from the high bits down.** `shift = bits * (per_word - 1 - col %
  per_word)` is the big-endian reading of the word. Every group of eight weights is
  reversed; a model loaded this way produces noise.
- **A group index without the row** (`col / group_size`). Every row uses the scales
  and biases of row 0. Right for a one-row test.
- **A one-dimensional grid of `in` threads** with a body that reads
  `thread_position_in_grid.y`. The y coordinate is always 0, so only row 0 is
  written and the other rows hold whatever was in the buffer.
- **A signed shift** (`int`, two shifts). Codes of 8 and above at 4 bits, 128 and
  above at 8 bits, come out negative.
- **A hard-coded mask** (`& 0xF`). Right for 4 bits; 8-bit codes lose their upper
  half and 2-bit codes pick up their neighbor.
- **Always writing float32.** A layer stored with bfloat16 scales comes back at twice
  the size the model's dtype calls for.
- **Turning off `ensure_row_contiguous`** and keeping row-major indices. Wrong as
  soon as an input is a strided view.
- **One thread per row**, or one kernel launch per row from Python. Correct, and 7
  and 80 times slower than one thread per weight.
- **MLX operations, or `mx.dequantize`.** Right numbers, no kernel; the first also
  triples the peak memory.

## Complexity

`out * in` threads, each with three reads and one write. The output is the only
allocation.

## In the wild

`mx.quantized_matmul` never builds this matrix: it unpacks codes inside the
multiplication, which is why a quantized model runs in the memory of its packed
weights. The float matrix is needed when weights are edited rather than used: merging
an adapter, re-quantizing to another bit width, inspecting a layer. Other formats
differ in exactly the lines you wrote: GGUF's `Q4_K` packs 4-bit codes in blocks of
32 with 6-bit scales per sub-block, and bitsandbytes stores 4-bit indices into a
16-entry table. The thread-per-weight structure is the same; only the index
arithmetic changes.
