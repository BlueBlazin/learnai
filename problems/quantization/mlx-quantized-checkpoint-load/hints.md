You need three byte ranges out of 3.9 GB, and the only thing that knows where they
are is the header at the front of the file. How many bytes do you have to look at
before you know all three ranges, and what is the smallest set of slices that gets
you the data? Then: a `U32` tensor of shape `(512, 640)` stands for a matrix with
2560 columns. Where did the factor of four go?

---

1. `n = int.from_bytes(buf[0:8], "little")`, `table = json.loads(bytes(buf[8:8 + n]))`,
   `data_start = 8 + n`. That is the only slice besides the tensors.
2. Look up `name + ".weight"`, `".scales"`, `".biases"` in `table` (a missing one is
   your `KeyError`), and check the shapes against `bits` and `group_size`.
3. For each, `np.frombuffer(buf[data_start + begin : data_start + end], dtype)` and a
   reshape: `"<u4"` for the words, `"<u2"` for bfloat16 patterns, `"<f2"` or `"<f4"`
   for the other float types.
4. Unpack the codes, convert scales and biases to float32, apply them per group.

Keep every offset a Python `int`.

---

Unpacking for any `bits`, with `p = 32 // bits` codes per word:

```python
shifts = np.arange(p, dtype=np.uint32) * np.uint32(bits)
codes = (words[:, :, None] >> shifts) & np.uint32(2**bits - 1)  # (out, n_words, p)
```

Flattening the last two axes gives the codes in input order. Reshape them to
`(out, in // group_size, group_size)`, convert to float32, and broadcast the scales
and biases as `[:, :, None]`:

```python
w = (codes * scale[:, :, None] + bias[:, :, None]).reshape(out, in_dim)
```

A bfloat16 pattern `u` (uint16) is the float32 with the bits `u << 16`:
`(u.astype(np.uint32) << 16).view(np.float32)`, or `bf16_to_f32(u)` from the kit.
