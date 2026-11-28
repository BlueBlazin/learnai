## The idea

This problem stacks two formats, each built around a single clean design choice.

**safetensors puts the index at the front.** The first 8 bytes give the header length $n$, and the JSON header maps every tensor name to its byte range relative to byte $8 + n$. Reading $8 + n$ bytes tells you where everything lives, so unrequested tensors on a memory-mapped file cost zero I/O and trigger no page faults. A layer's three tensors are not adjacent in the file. In the real checkpoint, layer 0's `k_proj` scales sit 245 MB before its codes, so reading a contiguous chunk around the layer will not work. Taking three exact slices does.

**MLX packs codes into 32-bit words, lowest bits first.** A `U32` tensor of shape `(out, in * bits / 32)` stores `out` rows of codes with `32 / bits` codes per word. At 8 bits in little-endian order, each byte is already one code in column order. At 4 bits or 2 bits, however, code 0 occupies the lowest bits of each word rather than the highest. Shifting by a small array of bit offsets and masking handles every bit width uniformly.

The final step inverts `quantize-real-weights`. Each consecutive run of `group_size` codes in a row shares a scale and a bias stored in the model's floating-point format (`BF16` for Granite). Because `BF16` is simply `float32` truncated to its upper 16 bits, decoding it is an exact left shift by 16 bits.

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

For the real checkpoint's `k_proj` layer, this solution slices 105,456 bytes for the header and 1,392,640 bytes for the three tensors (1.5 MB out of 3.9 GB) in a few tens of milliseconds.

## Common mistakes

- **Decoding words big-endian or taking the first code from the high bits.** Both reverse the code order within each 32-bit word. At 8 bits, this scrambles every block of four columns.
- **Unpacking the high nibble first.** Reading bytes and splitting each byte from the top works at 8 bits but fails at 4 bits and 2 bits.
- **Using signed codes.** Interpreting 8-bit codes as `int8` turns every code above 127 negative. Codes are always unsigned, and sign comes from the scale and bias.
- **Tiling scales instead of repeating them.** Using `np.tile` or reshaping to `(out, group_size, n_groups)` assigns input `j` to group `j % n_groups` instead of consecutive runs of `group_size` inputs.
- **Confusing `BF16` with `F16`.** Both are 16-bit formats, so `0x3F00` decodes to `0.5` in `BF16` and `1.75` in `F16`. Always dispatch on the header's `dtype` field.
- **Treating `data_offsets` as file offsets.** Offsets are relative to `8 + header_len`, not byte 0. Forgetting `data_start` reads JSON header text and neighboring tensors.
- **Computing `(code + bias) * scale`.** The scale multiplies only the code: `code * scale + bias`.
- **Hardcoding `bits = 8` or `group_size = 64`.** Skipping shape checks against the arguments causes a 4-bit checkpoint to silently return a matrix of the wrong width instead of raising `ValueError`.
- **Accepting unquantized tensors.** `model.norm.weight` exists in the header, but `model.norm` has no `.scales` or `.biases` and must raise `KeyError`.
- **Returning `float64`.** Multiplying codes by a float64 scale produces a float64 array, doubling memory usage relative to float32.
- **Reading the entire file.** Calling `bytes(buf[:])` loads 3.9 GB of data just to extract 1.4 MB.
- **Storing file offsets in 32-bit integers.** Layer 20's tensors start at bytes 2.5 billion and 3.6 billion, which overflows `int32` on the real checkpoint after passing all small tests. Keep offsets as 64-bit integers (Python `int`).

## Complexity

Parsing the header takes $O(n)$ time and memory, and dequantizing one layer takes $O(\text{out} \cdot \text{in})$ time, independent of the total file size. The unpacked `codes` array is the largest temporary at 4 bytes per weight when stored as `uint32` (unpacking into `uint8` reduces it to 1 byte per weight).

## In the wild

`mx.load` uses this same header-driven slicing lazily across all tensors. During inference, `nn.QuantizedLinear` avoids full dequantization altogether. It keeps the packed weights, scales, and biases in memory and calls `mx.quantized_matmul`, which unpacks codes on the fly inside the kernel so the 3B model uses 3.9 GB instead of 14.6 GB. Dequantizing one layer at a time, as done here, is how inspection and analysis tools work. `llama.cpp`'s GGUF format makes the same two architectural choices (an upfront tensor index and block-wise packed codes with per-block scales) with different binary details.
