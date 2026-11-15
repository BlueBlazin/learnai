Three numbers describe the same memory: what tensors occupy, what the allocator holds,
and what was the most tensors ever occupied. For each question, which of the three is
it about? Is memory that the allocator holds and no tensor occupies "in use" when you
are about to allocate more through the same allocator? And if you only look at the
allocated memory before and after a function, what happens to a tensor the function
created and deleted in between?

---

`predicted_allocated`: the number of elements is the product of the shape
(`math.prod`, which gives 1 for the empty tuple), the size of an element is
`dtype.itemsize`, and each tensor's bytes are rounded up to a multiple of 512 before
they are added. `fragmentation` and `kv_budget`: read the keys with
`stats.get(key, 0)`. The free memory for the budget is the total minus the *peak* of
*allocated* bytes; apply the safety fraction to that difference, floor, and clamp at
zero. `measure_peak`: reset the peak, read the allocated memory, call `fn`, read the
peak.

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
