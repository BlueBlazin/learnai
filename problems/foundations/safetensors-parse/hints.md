Three byte positions matter in a safetensors file: the start of the file (byte 0), the
start of the JSON header (byte 8), and the start of the data section. Which of those do
`data_offsets` count from, and how can you find that position before parsing the JSON?

---

In `read_header`, read the first 8 bytes as a little-endian unsigned 64-bit integer $n$
with `int.from_bytes(..., "little")` or `struct.unpack("<Q", ...)`. Verify that the
buffer has at least $8 + n$ bytes, then pass bytes `8` through `8 + n` to `json.loads`
(trailing spaces are valid JSON). Pop `"__metadata__"` from the dict so only tensor
entries remain, and return `8 + n` as `data_start`.

In `read_tensor`, look up `name`, map `dtype` using a table of little-endian NumPy dtype
strings (`"BF16": "<u2"`), and verify that `end - begin` equals the element count times
`dtype.itemsize`. To build the array without copying, avoid slicing a `bytes` object
(`buf[a:b]`) and instead tell NumPy to start at a byte offset inside `buf`.

---

```python
count = math.prod(info["shape"])  # 1 for shape [], 0 if any dimension is 0
flat = np.frombuffer(buf, dtype=dtype, count=count, offset=data_start + begin)
return flat.reshape(info["shape"])
```

`np.frombuffer` wraps any object supporting the buffer protocol (`bytes`, `bytearray`,
`memoryview`, `mmap`) without copying, and `reshape` on a contiguous array returns a view
as well. Use `len(memoryview(buf))` to check the buffer size, and raise `ValueError` for
an unknown `dtype` before indexing your dtype table.
