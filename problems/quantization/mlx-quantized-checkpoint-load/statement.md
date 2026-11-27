Implement `load_quantized` to read and dequantize a single layer from a real MLX quantized safetensors checkpoint, such as the 3.9 GB 8-bit build of IBM's Granite 4.2 3B. Your function must locate the layer's three tensors in the file header, slice only the required bytes from the file, unpack the integer codes from 32-bit words, decode the scales and biases, and reconstruct the float32 weight matrix.

## Specification

```python
def load_quantized(buf, name, bits, group_size) -> np.ndarray: ...
```

- `buf`: the checkpoint file as an object supporting `len(buf)` and slicing. `buf[a:b]` returns bytes `a` through `b - 1` as a bytes-like object (`bytes` or `memoryview`) accepted by `np.frombuffer`, `bytes(...)`, and `int.from_bytes`. `buf` may be a `bytes` object, an `mmap.mmap`, or the kit's [`CountingFile`](../../../python/src/learnai/kit/mlxtools.py), which tracks the total bytes sliced from a memory-mapped file. Read from `buf` using slices only.
- `name`: the layer prefix without a tensor suffix, such as `"model.layers.0.self_attn.k_proj"`.
- `bits`: bits per integer code (`2`, `4`, or `8`).
- `group_size`: number of consecutive inputs sharing one scale and one bias. Both `bits` and `group_size` come from the `quantization` entry in `config.json` (for this checkpoint, `{"group_size": 64, "bits": 8}`).
- Returns the dequantized weight matrix as a `float32` NumPy array of shape `(out, in)`.

### File layout

The file uses the safetensors format (as in `safetensors-parse`). Bytes 0 through 7 store the header length $n$ as an unsigned 64-bit little-endian integer. Bytes 8 through $8 + n - 1$ contain a JSON object mapping each tensor name to `{"dtype", "shape", "data_offsets": [begin, end]}`. Each tensor occupies bytes `begin` through `end - 1` relative to the start of the data section at byte $8 + n$. Tensors are stored in little-endian row-major order and may appear in any order in the file. The `"__metadata__"` key is not a tensor.

### Quantized layer tensors

Every quantized layer is stored as three tensors:

| tensor | dtype | shape | content |
|---|---|---|---|
| `{name}.weight` | `U32` | `(out, in * bits / 32)` | packed codes, `32 / bits` per word |
| `{name}.scales` | `BF16`, `F16`, or `F32` | `(out, in / group_size)` | one scale per group |
| `{name}.biases` | same as scales | `(out, in / group_size)` | one bias per group |

### Unpacking codes

Row $i$ of `{name}.weight` stores the `in` codes for output $i$, packed lowest bits first into 32-bit unsigned words. With $p = 32 / \text{bits}$ codes per word, code $j$ of row $i$ is

$$
c_{i,j} = \Big\lfloor \frac{\text{word}_{i,\,\lfloor j / p \rfloor}}{2^{\,\text{bits} \cdot (j \bmod p)}} \Big\rfloor \bmod 2^{\text{bits}},
$$

which is an unsigned integer in $[0, 2^{\text{bits}} - 1]$. Because each word is stored little-endian, at 8 bits the raw bytes of `{name}.weight` in file order are already the codes in column order.

### Scales, biases, and dequantization

Convert `{name}.scales` and `{name}.biases` to `float32` before applying them. A `BF16` value holds the upper 16 bits of a float32 number, so decoding places those 16 bits in the high half of a 32-bit word with zeros in the low half. You may call the kit's `bf16_to_f32(patterns)` on a `uint16` array for this conversion. `F16` is IEEE 754 half precision.

Group $g$ of row $i$ covers inputs $g \cdot \texttt{group\_size} \le j < (g + 1) \cdot \texttt{group\_size}$. Compute each weight in `float32` as

$$
w_{i,j} = c_{i,j} \cdot \text{scales}_{i,\,\lfloor j / \texttt{group\_size} \rfloor} + \text{biases}_{i,\,\lfloor j / \texttt{group\_size} \rfloor}.
$$

Scales may be negative.

### Errors

- Raise `KeyError` if any of the three tensors (`{name}.weight`, `{name}.scales`, `{name}.biases`) is missing from the header. For example, an unquantized norm gain such as `model.norm.weight` has no `.scales` or `.biases`, so `load_quantized(buf, "model.norm", ...)` must raise `KeyError`.
- Raise `ValueError` if `{name}.weight` does not have dtype `U32`, if `in` (words per row times `32 / bits`) is not divisible by `group_size`, if `{name}.scales` or `{name}.biases` does not have shape `(out, in / group_size)`, or if their dtype is not one of `BF16`, `F16`, or `F32`.

### Byte budget and forbidden calls

The total bytes sliced from `buf` (the sum of `b - a` over all slices) must not exceed the size of the layer's three tensors plus **three times the header size** ($3 \cdot (8 + n)$). Parse the header first, then slice only the three tensors. Operations that copy the entire buffer, such as `buf[:]`, `bytes(buf)`, or `buf.read()`, are charged for every byte they return.

Read the file and dequantize yourself. Do not use the `safetensors` package, the kit's `SafetensorsFile`, `RealModel`, `dequantize`, or `unpack_codes`, or MLX's `mx.load` and `mx.dequantize`.

## Example

Consider a 264-byte file containing one layer `proj` with 1 output and 8 inputs, quantized at 8 bits with `group_size = 4`:

```
offset  bytes
     0  f0 00 00 00 00 00 00 00                          n = 240
     8  {"__metadata__":{"format":"mlx"},
        "proj.weight":{"dtype":"U32","shape":[1,2],"data_offsets":[0,8]},
        "proj.scales":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]},
        "proj.biases":{"dtype":"BF16","shape":[1,2],"data_offsets":[12,16]}}
        (233 bytes of JSON on one line, then 7 spaces)
   248  00 01 02 03  05 0a 80 ff                         proj.weight: two words
   256  00 3f  80 be                                     proj.scales
   260  80 bf  00 40                                     proj.biases
```

```python
load_quantized(buf, "proj", bits=8, group_size=4)
# array([[-1.0, -0.5, 0.0, 0.5, 0.75, -0.5, -30.0, -61.75]], dtype=float32)
```

The two little-endian `U32` words are `0x03020100` and `0xFF800A05`. Unpacked lowest byte first, they yield the unsigned codes `0, 1, 2, 3` and `5, 10, 128, 255`. The scales are the bfloat16 patterns `0x3F00` ($0.5$) and `0xBE80` ($-0.25$), and the biases are `0xBF80` ($-1.0$) and `0x4000` ($2.0$). The first group computes $c \cdot 0.5 - 1.0$, and the second computes $c \cdot (-0.25) + 2.0$. Code `128` decodes to $-30.0$ and code `255` decodes to $-61.75$. If the bytes were read as signed `int8`, those two codes would be $-128$ and $-1$.

At 4 bits, eight codes fit in a single 32-bit word. For example, the codes `0, 1, 2, 3, 5, 10, 8, 15` pack into `0xF8A53210`, reading the hex digits from right to left.

In the real checkpoint, `model.layers.0.self_attn.k_proj.weight` has shape `[512, 640]`, which corresponds to 512 outputs, $640 \cdot 4 = 2560$ inputs, and 40 groups of 64 per row. Its three tensors total 1,392,640 bytes spread across 277 MB of the file, and the header is 105,448 bytes.

## Constraints

- Outputs must be `float32` NumPy arrays matching MLX's `mx.dequantize` (on float32 scales and biases) to a relative tolerance of `1e-6` and an absolute tolerance of `1e-7`.
- Tests that do not require a download use synthetic files and two small checkpoints written by `mx.save_safetensors`: 8-bit codes in groups of 64 with `BF16` scales, and 4-bit codes in groups of 32 with `F16` and `F32` scales. The byte budget applies to all of them.
- The real checkpoint is `model.safetensors` from `lmstudio-community/granite-4.2-3b-MLX-8bit`, which is 3,888,771,048 bytes and contains 927 tensors. Tests load three layers of shape `(512, 2560)`, each with a budget of 1.709 MB. Half of the file lies past byte $2^{31}$, so file offsets do not fit in 32-bit integers.
- `in` is at most 2560, so one layer fits easily in memory. The time limit is 10 seconds per case, and a correct implementation runs in a few milliseconds.

## Why this matters

Every "MLX 4-bit" and "MLX 8-bit" model on Hugging Face uses this exact file format, and a full model loader is just this function called in a loop. Implementing it directly lets you inspect checkpoints without loading a framework or an entire model into RAM. By combining a memory-mapped file with the safetensors header, you can read 1.4 MB from a 40 GB checkpoint on a laptop to verify quantization settings, compare layers across conversions, or gather weight statistics.

Bugs in checkpoint loaders tend to fail silently. Reading codes as signed bytes, decoding words as big-endian, or interpreting `BF16` as `F16` all produce a matrix of the right shape with plausible numbers, yielding a model that loads, runs, and outputs gibberish. Keeping file offsets in 32-bit integers is another classic trap. It passes every small test and only breaks on multi-gigabyte checkpoints.
