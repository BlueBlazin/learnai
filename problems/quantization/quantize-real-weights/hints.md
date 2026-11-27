Think about one question for each function. For `quantize_dequantize`, the textbook grid spans from a group's minimum to its maximum, so it hits both endpoints and usually misses zero. If you could shift one end of the grid slightly, which value would you want on the grid, and which endpoint can you afford to clip? For `quantize_report`, what does the quantized checkpoint store besides the integer codes, and which tensors in the model are not 2-D matrices?

---

For `quantize_dequantize`, reshape `w` to `(out, in // group_size, group_size)`, compute each step per group with `keepdim=True`, and reshape back at the end. Follow this order:

1. Compute `lo` and `hi` along the last axis, and `from_low = |lo| > |hi|`.
2. Compute `step = max((hi - lo) / L, 1e-7)`.
3. Set `scale = where(from_low, step, -step)` and `edge = where(from_low, lo, hi)`.
4. Compute `q0 = round(edge / scale)`. Where `q0 != 0`, replace `scale = edge / q0` and set `bias = edge`. Elsewhere, set `bias = 0`. Guard the denominator so the unused branch never divides by zero.
5. Compute `codes = clamp(round((w - bias) / scale), 0, L)` and return `codes * scale + bias`.

For `quantize_report`, iterate once over `weights`. If a tensor is not 2-D, add its element count to the parameter total and `16 * numel` to the bit total, then continue. If it is 2-D, quantize it, add its element count, its `bits * numel + 32 * numel // group_size` bits, its signal power, and its noise power to four running totals, and record its per-layer SQNR.

---

Keep the running totals as Python `float` and `int` values rather than float32 or int32 tensors, because the real model has $3.5 \times 10^8$ weights and occupies about $3 \times 10^9$ bits. Sum the squares of large matrices in two stages (`.sum(-1).sum()`) so float32 accumulation does not lose small squares against a large running total. Then compute the outputs directly:

```python
sqnr_db[name] = 10.0 * math.log10(signal / noise)
model = 10.0 * math.log10(signal_total / noise_total)  # pooled, not a mean of dB
bpw = stored_bits / parameters
worst = sorted(sqnr_db, key=sqnr_db.get)[:3]
```
