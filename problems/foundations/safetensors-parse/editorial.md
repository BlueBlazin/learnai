## The idea

A safetensors file is a length prefix, a JSON table of contents, and raw binary arrays.
Let $n$ be the unsigned 64-bit little-endian integer in the first 8 bytes, and let
`[begin, end]` be a tensor's `data_offsets` entry. Because `data_offsets` counts from the
start of the data section, the tensor's bytes occupy the file slice

$$
[\,8 + n + \text{begin},\; 8 + n + \text{end}\,) .
$$

Offsets are relative to the data section so a writer can lay out tensor data first and
write the header afterward without patching every offset.

To avoid copying, let NumPy point directly into `buf`. Calling
`np.frombuffer(buf, dtype, count, offset)` creates an array whose data pointer starts
`offset` bytes into the buffer's memory without reading or allocating tensor bytes. When
`buf` is an `mmap`, the operating system faults in pages only when elements are accessed
and can evict them under memory pressure because they are file-backed. That is how a
multi-gigabyte model loads almost instantly. Slicing a `bytes` object (`buf[a:b]`) or
calling `bytes(...)` on tensor data defeats this by allocating a full copy.

Copying the header itself with `bytes(view[8 : 8 + n])` is fine. In the Granite
checkpoint, the header is only 26 kB, and `json.loads` needs a contiguous byte string
anyway.

`BF16` tensors are returned as `uint16` because NumPy has no `bfloat16` dtype. Each
16-bit value is the upper half of a `float32` bit pattern, so
`(bits.astype(np.uint32) << 16).view(np.float32)` decodes them as in the previous
problem.

## Reference solution (Python)

```python
DTYPES = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "U16": "<u2",
    "I16": "<i2",
    "U32": "<u4",
    "I32": "<i4",
    "U64": "<u8",
    "I64": "<i8",
    "F16": "<f2",
    "BF16": "<u2",
    "F32": "<f4",
    "F64": "<f8",
}


def read_header(buf):
    view = memoryview(buf)
    if len(view) < 8:
        raise ValueError("a safetensors file starts with an 8-byte header length")
    n = int.from_bytes(view[:8], "little")
    if n > len(view) - 8:
        raise ValueError("the header is longer than the file")
    tensors = json.loads(bytes(view[8 : 8 + n]))
    metadata = tensors.pop("__metadata__", None) or {}
    return tensors, metadata, 8 + n


def read_tensor(buf, name):
    tensors, _, data_start = read_header(buf)
    info = tensors[name]
    if info["dtype"] not in DTYPES:
        raise ValueError(f"unknown dtype {info['dtype']!r}")
    dtype = np.dtype(DTYPES[info["dtype"]])
    begin, end = info["data_offsets"]
    count = prod(info["shape"])
    if not 0 <= begin <= end or end - begin != count * dtype.itemsize:
        raise ValueError("data_offsets do not fit the shape")
    if data_start + end > len(memoryview(buf)):
        raise ValueError("the data ends past the end of the file")
    flat = np.frombuffer(buf, dtype=dtype, count=count, offset=data_start + begin)
    return flat.reshape(info["shape"])
```

`math.prod([])` returns 1, which is the element count of a scalar tensor. If any
dimension in `shape` is 0, `prod` returns 0 and `np.frombuffer` returns an empty array.

## Common mistakes

- **Counting offsets from byte 0.** Reading `buf[begin:end]` returns a tensor of the
  right shape filled with header bytes. For the first tensor of the Granite checkpoint,
  that reads the header length and `{"__metadata__":...` as `bfloat16`.
- **Reading the header length as big-endian.** `struct.unpack(">Q", ...)` turns a
  152-byte header into $152 \cdot 2^{56}$ bytes. Every integer and float in safetensors
  is little-endian.
- **Using big-endian element dtypes.** Passing `">f4"` or `"!f"` produces finite garbage.
  On little-endian machines `np.float32` happens to equal `"<f4"`, but specifying `<`
  explicitly is portable everywhere.
- **Treating `"__metadata__"` as a tensor.** Leaving it in `tensors` inflates the tensor
  count by one and crashes loops with `KeyError: 'dtype'`. Remember that `"__metadata__"`
  can appear anywhere in the JSON object, not only first.
- **Copying the tensor bytes.** Calling `np.frombuffer(buf[a:b])` on a `bytes` object,
  `bytes(view[a:b])`, `np.array(...)`, or `.copy()` produces the right values but doubles
  memory use (205 MB for the Granite embedding table alone). Calling `.astype(...)` also
  copies by default.
- **Decoding `BF16` as `np.float16`.** IEEE half precision has 5 exponent bits, while
  `bfloat16` has 8. The bit pattern `0x3F80` is $1.0$ in `bfloat16` but $1.875$ in
  `float16`.
- **Assuming data is stored in header order.** Writers arrange tensors however they like
  (the Rust library sorts by dtype alignment and name, while MLX uses another order), and
  JSON objects are unordered. Always use `data_offsets` rather than accumulating sizes.
- **Ignoring `end` and trusting `shape`.** Checking `end - begin` against
  `count * dtype.itemsize` and verifying `data_start + end <= len(buf)` is the format's
  integrity check against truncated downloads or mismatched headers.
- **Reshaping in column-major order.** Tensors are stored in C (row-major) order. Passing
  `order="F"` scrambles non-square matrices and transposes square ones.

## Complexity

`read_header` takes $O(n)$ time and $O(n)$ memory in the header length $n$. `read_tensor`
adds $O(1)$ time and memory on top of parsing the header because no tensor bytes are read
or copied.

## In the wild

The official `safetensors` library is a Rust crate with Python bindings. Its `safe_open`
memory-maps the file in the same way, and `safetensors.torch.load_file` builds tensors on
top of that map. Production loaders also enforce stricter checks than this problem
requires. The header is capped at 100 MB so a corrupt length cannot trigger a massive
allocation, and tensor byte ranges must tile the data section with no gaps or overlaps to
prevent aliased or hidden data.

Security is the format's other main motivation. Unlike pickle-based `.bin` or `.pt`
checkpoints, parsing a safetensors file executes no code and can only produce arrays.
Notice also that tied weights are stored only once, so the Granite checkpoint has no
`lm_head.weight` entry. That detail returns in `real-model-sizing` and
`real-model-logits`.
