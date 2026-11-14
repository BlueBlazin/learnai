Two questions, one per function. For the quantizer: the textbook grid runs from the
minimum to the maximum of a group, so it contains both ends and, in general, not
zero. If you could move one end of the grid a little, which value would you most want
on it, and which end can you afford to lose? For the report: what does the file on
disk contain besides the codes, and what in the model is not a matrix?

---

`quantize_dequantize`: reshape to `(out, in // group_size, group_size)` and compute
everything per group with `keepdim`, then reshape back. In order: `lo`, `hi`;
`from_low = |lo| > |hi|`; `step = max((hi - lo) / L, 1e-7)`;
`scale = where(from_low, step, -step)`; `edge = where(from_low, lo, hi)`;
`q0 = round(edge / scale)`; where `q0 != 0`, `scale = edge / q0` and `bias = edge`,
elsewhere `bias = 0`; `codes = clamp(round((w - bias) / scale), 0, L)`; return
`codes * scale + bias`. Guard the division by `q0` so that the unused branch does not
divide by zero.

`quantize_report`: one loop over `weights`. For a tensor that is not 2-D, add its
size to the parameter count and 16 bits per element to the bit count, and move on.
For a matrix, add its size, its `bits * size + 32 * size / group_size` bits, its
signal power and its noise power to four running totals, and store its SQNR.

---

Keep the running totals as Python numbers (`float(...)`, `int`), not as float32 or
int32 tensors: the real model has 3.5e8 weights and occupies 3e9 bits. Sum the
squares of a large matrix in two stages (`.sum(-1).sum()`), so that the float32 sum
never adds a tiny square to a huge partial sum. Then

```python
sqnr_db[name] = 10.0 * math.log10(signal / noise)
model = 10.0 * math.log10(signal_total / noise_total)  # pooled, not a mean of dB
bpw = stored_bits / parameters
worst = sorted(sqnr_db, key=sqnr_db.get)[:3]
```
