Work in bytes and follow the order in which memory is claimed. The reserve is a share of the whole device, not of what remains after loading the weights, and the weights do not shrink when you set memory aside.

---

Compute one shared helper value: the number of whole blocks in the pool. Round the reserve up, subtract the weights, divide by the bytes per block (rounding down), and clamp the result at `0` so negative budgets do not become negative block counts under Python's `//`. You can compute an integer ceiling without floats as `(a + b - 1) // b` or `-(-a // b)`.

---

```python
reserved = (memory_bytes * overhead_percent + 99) // 100
cache_bytes = memory_bytes - reserved - weight_bytes
block_bytes = block_tokens * 2 * n_layers * n_kv_heads * head_dim * itemsize
blocks = max(0, cache_bytes // block_bytes)
```

From `blocks`, compute `max_sequences` as `blocks // ceil(context / block_tokens)` and `max_context` as `(blocks // batch) * block_tokens`. On the worked example, you should get 12,110 blocks, 256 blocks per 4,096-token sequence, and 47 sequences.
