## The idea

Two formats are stacked here, and each has one idea.

**safetensors puts the map in front.** The first 8 bytes say how long the header is,
and the header says where every tensor lies, relative to the end of the header. So a
reader looks at $8 + n$ bytes and then knows everything; the tensors it does not want
cost nothing, not even a page fault, if the file is memory-mapped. The three tensors
of one layer are not neighbors in the file (in the real checkpoint the scales of
layer 0's `k_proj` lie 245 MB before its codes), so
"read a chunk around it" does not work. Three exact slices do.

**MLX packs codes into 32-bit words, lowest bits first.** A `U32` tensor of shape
`(out, in * bits / 32)` is the codes of `out` rows, `32 / bits` per word. At 8 bits
and little-endian storage this is simply one code per byte, in order, which is why
the format looks trivial until the first 4-bit file, where code 0 of a word is its
low nibble. A shift-and-mask over a small array of shifts handles every width.

The arithmetic at the end is the inverse of `quantize-real-weights`: every run of
`group_size` consecutive codes in a row shares a scale and a bias, both stored in the
model's floating-point type, bfloat16 for Granite. bfloat16 is a float32 with the low
16 bits cut off, so decoding it is a shift, and it is exact.

## Reference solution (Python)

```python
FLOAT_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}


def _read(buf, info, data_start, dtype):
    begin, end = info["data_offsets"]
    flat = np.frombuffer(buf[data_start + begin : data_start + end], dtype=dtype)
    return flat.reshape(info["shape"])


def _read_floats(buf, info, data_start):
    if info["dtype"] not in FLOAT_DTYPES:
        raise ValueError(f"scales and biases must be F32, F16, or BF16, not {info['dtype']}")
    stored = _read(buf, info, data_start, FLOAT_DTYPES[info["dtype"]])
    return bf16_to_f32(stored) if info["dtype"] == "BF16" else stored.astype(np.float32)


def load_quantized(buf, name, bits, group_size):
    header_len = int.from_bytes(buf[0:8], "little")
    table = json.loads(bytes(buf[8 : 8 + header_len]))
    data_start = 8 + header_len
    weight, scales, biases = (table[f"{name}.{part}"] for part in ("weight", "scales", "biases"))

    per_word = 32 // bits
    out_dim, n_words = weight["shape"]
    in_dim = n_words * per_word
    groups = [out_dim, in_dim // group_size]
    if weight["dtype"] != "U32" or in_dim % group_size:
        raise ValueError(f"{name}.weight does not hold {bits}-bit codes in groups of {group_size}")
    if scales["shape"] != groups or biases["shape"] != groups:
        raise ValueError(f"{name}: scales and biases must have shape {groups}")

    words = _read(buf, weight, data_start, "<u4")
    shifts = np.arange(per_word, dtype=np.uint32) * np.uint32(bits)
    codes = (words[:, :, None] >> shifts) & np.uint32(2**bits - 1)
    codes = codes.reshape(out_dim, in_dim // group_size, group_size).astype(np.float32)
    scale = _read_floats(buf, scales, data_start)[:, :, None]
    bias = _read_floats(buf, biases, data_start)[:, :, None]
    return (codes * scale + bias).reshape(out_dim, in_dim)
```

For the real `k_proj` it requests 105,456 bytes of header and 1,392,640 bytes of
tensors: 1.5 MB of 3.9 GB, in a few tens of milliseconds.

## Common mistakes

- **Big-endian words**, or **the first code from the high bits.** Both reverse the
  codes inside every word. At 8 bits the matrix has the right values in the wrong
  columns, four at a time.
- **High nibble first.** Unpacking bytes and then splitting each byte from the top is
  right at 8 bits and wrong below.
- **Signed codes.** `int8` turns every code above 127 into a negative number. Codes
  are unsigned, and the sign of a weight comes from the scale and the bias.
- **Tiled scales.** `np.tile` (or a reshape to `(out, group_size, n_groups)`) gives
  input `j` the scale of group `j % n_groups`. Groups are consecutive runs:
  `np.repeat`, or the reshape above.
- **bfloat16 read as float16**, or the reverse. Both are 16 bits wide; `0x3F00` is
  0.5 as bfloat16 and 1.75 as float16. The header's `dtype` decides.
- **Offsets from the start of the file.** They count from the data section. The
  result is a tensor of the right shape made of JSON text and neighboring weights.
- **`(code + bias) * scale`.** The scale applies to the code only.
- **Ignoring `bits` or `group_size`** because the checkpoint at hand uses 8 and 64.
  The next file does not, and a loader that does not check the shapes against its
  settings returns a matrix twice as wide instead of an error.
- **Accepting a tensor without scales.** `model.norm.weight` exists; `model.norm` is
  not a quantized layer.
- **float64.** The codes times a float64 scale give a float64 matrix, twice the size
  of the float32 model that quantization was supposed to shrink by four.
- **Reading the file.** `bytes(buf[:])` is 3.9 GB of I/O and memory to get 1.4 MB.
- **Offsets in int32.** The tensors of layer 20 start at bytes 2.5 and 3.6 billion.
  An offset table in a 32-bit integer array is fine for every test file and
  overflows on the real one. File offsets are 64-bit quantities.

## Complexity

$O(n)$ for the header and $O(\text{out} \cdot \text{in})$ for the layer, with no
dependence on the size of the file. The unpacked codes are the largest temporary:
4 bytes per weight as `uint32` here (unpack to `uint8` to make it 1).

## In the wild

`mx.load` does the same lazily for all tensors, and `nn.QuantizedLinear` never
dequantizes: it keeps the three tensors and calls `mx.quantized_matmul`, which
unpacks inside the kernel, so the 3B model occupies 3.9 GB in memory and not 14.6.
Dequantizing one layer at a time, as here, is what analysis tools do. llama.cpp's
GGUF makes the same two decisions with other details: a header that maps every
tensor, and block-wise codes with per-block scales.
