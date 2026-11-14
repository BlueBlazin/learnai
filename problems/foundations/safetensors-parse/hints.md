There are three different origins in play: the start of the file, the start of the
header (byte 8), and the start of the data section. Which of them do the numbers in
`data_offsets` count from, and how do you compute where that is before you have parsed
anything?

---

`read_header`: read the first 8 bytes as a little-endian unsigned integer $n$
(`int.from_bytes(..., "little")` or `struct.unpack("<Q", ...)`), check that the buffer
really has $8 + n$ bytes, and give bytes 8 to $8 + n$ to `json.loads` (trailing spaces
are fine for JSON). Pop `"__metadata__"` out of the resulting dict; what remains is the
tensor table, and `data_start` is $8 + n$.

`read_tensor`: look the name up, translate the dtype with a table of little-endian
numpy dtype strings (`"BF16": "<u2"`), and check `end - begin` against the element
count times the item size. Then build the array **on top of** the buffer. Slicing a
`bytes` object (`buf[a:b]`) makes a copy, so do not slice; ask numpy to start reading
at an offset instead.

---

```python
count = math.prod(info["shape"])  # 1 for shape [], 0 if any dimension is 0
flat = np.frombuffer(buf, dtype=dtype, count=count, offset=data_start + begin)
return flat.reshape(info["shape"])
```

`np.frombuffer` wraps the memory of anything that supports the buffer protocol
(`bytes`, `bytearray`, `memoryview`, `mmap`) without copying it, and `reshape` of a
contiguous array is a view too. Use `len(memoryview(buf))` for the size of the buffer,
and remember that the `ValueError` for an unknown dtype must come before you index
your dtype table.
