You need three byte ranges out of a 3.9 GB file, and only the header at the start of the file knows where they live. How many bytes must you read before you know all three ranges, and what is the smallest set of slices that fetches the layer? Also, a `U32` weight tensor of shape `(512, 640)` represents a matrix with 2560 columns—where did the factor of four go?

---

1. Read the header with two small slices: `n = int.from_bytes(buf[0:8], "little")` and `table = json.loads(bytes(buf[8 : 8 + n]))`, and set `data_start = 8 + n`.
2. Look up `name + ".weight"`, `name + ".scales"`, and `name + ".biases"` in `table` (a missing key naturally raises `KeyError`), then validate their dtypes and shapes against `bits` and `group_size`.
3. For each tensor, slice `buf[data_start + begin : data_start + end]`, wrap it with `np.frombuffer(..., dtype=...)`, and reshape to `info["shape"]`. Use `"<u4"` for words, `"<u2"` for `BF16`, `"<f2"` for `F16`, and `"<f4"` for `F32`.
4. Unpack the codes, convert scales and biases to `float32`, and apply them per group.

Keep every byte offset as a Python `int` so offsets past $2^{31}$ do not overflow.

---

To unpack any `bits` with `p = 32 // bits` codes per 32-bit word:

```python
shifts = np.arange(p, dtype=np.uint32) * np.uint32(bits)
codes = (words[:, :, None] >> shifts) & np.uint32(2**bits - 1)  # (out, n_words, p)
```

Merging the last two axes puts the codes in input order. Reshape them to `(out, in // group_size, group_size)`, cast to `float32`, and broadcast the scales and biases with `[:, :, None]`:

```python
w = (codes * scale[:, :, None] + bias[:, :, None]).reshape(out, in_dim)
```

A `uint16` array `u` of `BF16` bit patterns converts to `float32` via `(u.astype(np.uint32) << 16).view(np.float32)`, or by calling `bf16_to_f32(u)` from the kit.
