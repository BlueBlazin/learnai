## The idea

The format is a length, a JSON table of contents, and raw arrays. All the information
needed to find a tensor is in its header entry, and the only arithmetic is one
addition: a tensor's bytes are at

$$
[\,8 + n + \text{begin},\; 8 + n + \text{end}\,)
$$

in the file, where $n$ is the little-endian integer in the first 8 bytes. Offsets are
relative to the data section so that a writer can lay out the data first and write the
header afterwards without patching every entry.

Reading is then a question of *not doing work*. `np.frombuffer(buf, dtype, count,
offset)` creates an array object whose data pointer is `offset` bytes into the
buffer's memory. Nothing is read. If the buffer is an `mmap`, the operating system
loads a page of the file the first time an element on it is touched, and can drop it
again under memory pressure, because the page is backed by the file. That is the
whole trick behind "loading" a multi-gigabyte model instantly, and it is why the
function must never slice a `bytes` object or call `bytes(...)` on the data: both
allocate and copy.

The header itself *is* copied (`bytes(view[8 : 8 + n])`), which is fine: it is 26 kB
for the Granite checkpoint, and the JSON parser needs a contiguous string anyway.

`BF16` is returned as `uint16` because numpy has no bfloat16. The bit patterns are
the top halves of float32 patterns, so `(bits.astype(np.uint32) << 16).view(np.float32)`
decodes them, which is the previous problem.

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

`math.prod([])` is 1, which is the right element count for a scalar, and a shape with
a zero in it gives 0, for which `frombuffer` returns an empty array.

## Common mistakes

- **Offsets from the start of the file.** `buf[begin:end]` returns a tensor of the
  right shape made of the header's own bytes. For the first tensor of the Granite
  file that is the length, then `{"__metadata__":...`, read as bfloat16.
- **A big-endian length.** `struct.unpack(">Q", ...)` turns a 152-byte header into
  one of $152 \cdot 2^{56}$ bytes. Everything in the format is little-endian.
- **Big-endian elements.** The same mistake in the data (`">f4"`, or `"!f"` out of
  networking habit) gives finite garbage. On a little-endian machine `np.float32` is
  already `"<f4"`, but saying so costs nothing and is correct everywhere.
- **Treating `__metadata__` as a tensor.** The tensor count is off by one and any
  loop over the table dies on `KeyError: 'dtype'`. The key can be anywhere in the
  object, not only first.
- **Copying.** `np.frombuffer(buf[a:b])` on `bytes`, `bytes(view[a:b])`, `np.array(...)`,
  or `.copy()` all produce correct values and double the memory: 205 MB for the
  embedding table alone. A related trap is `.astype(...)`, which always copies unless
  told otherwise.
- **Decoding BF16 as half precision.** `np.float16` is IEEE half (5 exponent bits);
  bfloat16 has 8. `0x3F80` is $1.0$ as bfloat16 and $1.875$ as half precision.
- **Assuming the data is stored in header order.** Writers order the data to suit
  themselves (the Rust library sorts by dtype alignment and name; MLX files are in yet
  another order) and JSON objects are unordered by definition. Accumulating sizes
  works on the files you tested and fails on the next one. Use `data_offsets`.
- **Trusting the shape and ignoring `end`.** The redundancy between shape, dtype, and
  byte range is the format's integrity check. A truncated download or a header from
  another file is caught here or not at all.
- **Column-major reshape.** Tensors are row-major; `order="F"` scrambles every
  non-square matrix and transposes the square ones.

## Complexity

`read_header` is $O(n)$ in the header length. `read_tensor` adds $O(1)$ beyond that:
no tensor byte is read. Memory is $O(n)$ for the parsed header.

## In the wild

The reference implementation is a Rust crate with Python bindings; its `safe_open`
maps the file exactly like this, and `safetensors.torch.load_file` then builds tensors
on the map. It validates more than this problem asks: the header may be at most 100 MB
(so a corrupt length cannot make a loader allocate terabytes), and the byte ranges of
all tensors must tile the data section with no gaps and no overlaps, which rules out
files that alias or hide data. The absence of executable content is the format's other
reason to exist: unlike a pickle-based `.bin` or `.pt` file, parsing it can only ever
produce arrays. Tied weights are stored once, so the Granite file has no
`lm_head.weight`; that fact returns in `real-model-sizing` and `real-model-logits`.
