Work in bytes and in the order the memory is actually given away. Who takes their share
first, and of what total? The reserve is a share of the device, not of what is left
after the weights, and the weights do not shrink when you set memory aside.

---

Compute one intermediate quantity that both functions share: the number of whole
blocks in the pool. Reserve first (rounded up), then subtract the weights, then divide
by the bytes of one block and round down; clamp at zero, because Python's `//` on a
negative number rounds toward minus infinity and would give `-1`, not `0`. Integer
ceiling without floats is `-(-a // b)` or `(a + b - 1) // b`.

---

```python
reserved = (memory_bytes * overhead_percent + 99) // 100
cache_bytes = memory_bytes - reserved - weight_bytes
block_bytes = block_tokens * 2 * n_layers * n_kv_heads * head_dim * itemsize
blocks = max(0, cache_bytes // block_bytes)
```

Then `max_sequences` is `blocks // ceil(context / block_tokens)` and `max_context` is
`(blocks // batch) * block_tokens`. Check the example: 12,110 blocks, 256 blocks for a
4096-token sequence, 47 sequences.
