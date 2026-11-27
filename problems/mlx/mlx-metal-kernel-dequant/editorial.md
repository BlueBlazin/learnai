## The idea

Dequantization is embarrassingly parallel because every output weight depends only on its own code, scale, and bias. Assigning one GPU thread per output weight reduces the problem to mapping each thread's grid coordinates to its three input addresses.

For a matrix with `n_out` rows and `n_in` columns (`n_words` packed words and `n_groups` groups per row), launch a 2D grid with `grid=(n_in, n_out, 1)` so that `col = thread_position_in_grid.x` and `row = thread_position_in_grid.y`. Let `per_word = 32 / bits` be the number of codes per word. Each thread computes four indices:

| quantity | formula | reason |
|---|---|---|
| word index | `row * n_words + col / per_word` | `per_word` codes per word, stored row-major |
| bit shift | `bits * (col % per_word)` | code 0 sits in the lowest bits of each word |
| group index | `row * n_groups + col / group_size` | `scales` and `biases` have shape `(n_out, n_groups)` in row-major order |
| output index | `row * n_in + col` | the output matrix is row-major |

With those indices in hand, each thread evaluates `code * scale + bias`.

Three implementation details ensure exact results and high throughput:

- **Use unsigned integers everywhere.** Right-shifting a `uint` fills the high bits with zeros. Right-shifting a signed `int` replicates the sign bit. That bug hides if you mask afterward, but breaks immediately if you extract bit fields with two shifts (`(word << a) >> b`), turning every code whose top bit is 1 into a negative number.
- **Pass `bits` as a template integer.** Setting `template=[("BITS", bits)]` makes `32 / BITS` and the bitmask compile-time constants. The Metal compiler lowers divisions and modulos by `per_word` into bit shifts and masks instead of emitting integer division instructions in every thread.
- **Compute in `float` and convert once.** The output type `T` can be float16, bfloat16, or float32. Calling `fma(float(code), float(scale), float(bias))` rounds once in float32, and `T(...)` rounds once more to the output dtype. This matches `mx.dequantize` bit for bit across all three dtypes on Apple Silicon. Computing directly in `T` fails to compile for bfloat16 (MLX's Metal bfloat16 type supports storage and conversion, not arithmetic) and loses precision in float16 when multiplying an 8-bit code (up to 255) by a scale before adding the bias.

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

Reading row widths from `words_shape` and `scales_shape` lets one compiled kernel per `(dtype, bits)` pair handle every layer shape in a model. The largest flat index is `out * in`, which is 16.8 million for a `4096 x 4096` layer and 257 million for a `100352 x 2560` embedding table, both well within 32-bit `uint` range.

Leaving `ensure_row_contiguous` at its default (`True`) automatically copies any strided input into a contiguous buffer before launch, keeping flat row-major indexing valid. Packed checkpoint weights are already contiguous, so no extra copy occurs in normal inference.

## What the alternatives cost

Measured in steady state on an M1 Max (MLX 0.32.4) for a `4096 x 4096` layer at 4 bits with `group_size=64` (8 MiB of packed words and a 64 MiB float32 output):

| implementation | time per call | peak memory above inputs |
|---|---|---|
| this kernel, one thread per weight | 0.50 ms | 64 MiB |
| `mx.dequantize` | 0.43 ms | 64 MiB |
| MLX operations: shift, mask, reshape, cast, `repeat`, multiply, add | 2.8 ms | 192 MiB |
| a kernel with one thread per row, looping over columns | 3.7 ms | 64 MiB |
| the row kernel called from a Python loop, 4096 launches | 39 ms | 128 MiB |

The custom kernel matches MLX's built-in implementation. Composing standard MLX ops is more than 5x slower and triples peak memory because each step materializes a full-size intermediate array (unpacked uint32 codes, float codes, and repeated scales). Unlike simple elementwise chains, dequantization genuinely benefits from a fused custom kernel.

The exact threadgroup shape has little effect here. `(32, 8, 1)`, `(16, 16, 1)`, `(256, 1, 1)`, and `(1024, 1, 1)` all measure 0.50 ms. The decisive factor is spreading the work across 16.8 million short threads rather than 4096 long ones.

## Common mistakes

- **Reading codes from the high bits down.** Using `shift = bits * (per_word - 1 - col % per_word)` reverses every group of `per_word` weights, causing a loaded model to output nonsense.
- **Omitting the row from the group index.** Indexing scales and biases with `col / group_size` makes every row reuse row 0's parameters. This bug passes single-row tests.
- **Launching a 1D grid of `in` threads while reading `thread_position_in_grid.y`.** The y coordinate stays 0, so only row 0 is written and all other rows retain uninitialized buffer memory.
- **Using signed shifts (`int`).** Extracting codes with two signed shifts turns codes of 8 or higher at 4 bits (and 128 or higher at 8 bits) negative.
- **Hard-coding a 4-bit mask (`& 0xF`).** This works for 4 bits, truncates the upper half of 8-bit codes, and leaks neighboring bits into 2-bit codes.
- **Always outputting float32.** Ignoring `scales.dtype` doubles the memory footprint of a float16 or bfloat16 layer.
- **Disabling `ensure_row_contiguous` while using row-major flat indexing.** This breaks as soon as any input is a strided view.
- **Using one thread per row or one kernel launch per row.** Both produce correct values but run roughly 7x and 80x slower than one thread per weight.
- **Using standard MLX operations or `mx.dequantize`.** Both violate the single-custom-kernel rule, and the pure-MLX composition triples peak memory.

## Complexity

The kernel dispatches `out * in` threads, each performing three reads and one write. The output array is the only allocation.

## In the wild

During inference, `mx.quantized_matmul` never materializes the dequantized float matrix. It unpacks codes on the fly inside the matrix multiplication so the model stays within its packed memory footprint. Full dequantization is used when weights must be modified or inspected, such as merging LoRA adapters, re-quantizing to another bit width, or debugging a layer. Other quantization formats keep the same thread-per-weight structure and change only the index arithmetic. For example, GGUF's `Q4_K` packs 4-bit codes in blocks of 32 with 6-bit sub-block scales, while bitsandbytes stores 4-bit indices into a 16-entry lookup table.
