Three counters describe the same GPU memory: what live tensors occupy (`allocated`), what the allocator holds (`reserved`), and the highest value live tensors have reached (`peak allocated`). Ask which counter each function needs.

Is cached memory that no tensor occupies still "in use" when you allocate a KV cache through the same allocator? And if you only compare allocated memory before and after `fn()`, what happens to a temporary tensor created and freed in between?

---

- **`predicted_allocated`**: compute elements with `math.prod(shape)` (which returns `1` for `()`), multiply by `dtype.itemsize`, and round each tensor's byte count up to a multiple of 512 before summing.
- **`fragmentation` and `kv_budget`**: read keys with `stats.get(key, 0)`. For the KV budget, subtract `allocated_bytes.all.peak` from `total_bytes`, multiply by `1 - safety`, floor, and clamp at `0`.
- **`measure_peak`**: reset peak stats first, record the current allocated bytes, call `fn()`, and subtract the baseline from `max_memory_allocated()`.

---

```python
blocks = -(-nbytes // 512) * 512  # rounds up; 0 stays 0

free = total_bytes - stats.get("allocated_bytes.all.peak", 0)
budget = max(0, math.floor(free * (1.0 - safety)))

mem.reset_peak_memory_stats()
before = mem.memory_allocated()
fn()
peak = mem.max_memory_allocated() - before
```
