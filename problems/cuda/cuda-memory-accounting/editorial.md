## The idea

Requesting memory from the GPU driver is expensive, so PyTorch's caching allocator reserves large segments up front, carves blocks out of them for tensors, and keeps freed blocks cached for reuse. That design produces three distinct measurements of memory usage:

- **Allocated** (`torch.cuda.memory_allocated()`): bytes currently occupied by live tensors.
- **Reserved** (`torch.cuda.memory_reserved()`): bytes across all segments held by the allocator, whether occupied or cached. This is close to what `nvidia-smi` reports for the process, which also includes the CUDA context.
- **Peak allocated** (`torch.cuda.max_memory_allocated()`): the maximum allocated bytes since peak statistics were last reset.

In PyTorch 2.15, all three functions are single-key lookups into `torch.cuda.memory_stats()`: `allocated_bytes.all.current`, `reserved_bytes.all.current`, and `allocated_bytes.all.peak`.

Each function in this problem targets one of these quantities.

**Predicting allocated bytes.** The allocator rounds every allocation up to a multiple of 512 bytes. A 1-element tensor takes 512 bytes, and 100 of them take 51,200 bytes. Rounding must happen per tensor before summing, not once on the total.

**Measuring fragmentation.** Cached memory is reserved minus allocated ($R - A$). However, `empty_cache()` can only return completely free segments to the driver. A free block that shares its segment with a live tensor cannot be released. The allocator tracks these stuck bytes as `inactive_split_bytes` ($S$, printed as "Non-releasable memory" by `torch.cuda.memory_summary()`). A high ratio $S / R$ indicates fragmentation.

**Sizing the KV cache.** Because the KV cache is allocated through the same caching allocator, cached free blocks are available to it. What is unavailable is the memory occupied by live tensors at the busiest moment of a request: `allocated_bytes.all.peak`. Applying the safety margin to the remaining free memory leaves headroom for fragmentation and requests larger than the warm-up.

**Measuring a function's peak.** Peak allocated memory never decreases on its own, so you must reset peak stats before calling `fn`. Subtracting the baseline allocated memory from the new peak captures temporary allocations that were created and freed before `fn` returned.

## Reference solution (PyTorch)

```python
BLOCK = 512


def predicted_allocated(shapes, dtype):
    total = 0
    for shape in shapes:
        nbytes = math.prod(shape) * dtype.itemsize
        total += -(-nbytes // BLOCK) * BLOCK
    return total


def fragmentation(stats):
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    split = stats.get("inactive_split_bytes.all.current", 0)
    cached = reserved - allocated
    return {
        "cached_bytes": cached,
        "releasable_bytes": cached - split,
        "non_releasable_fraction": split / reserved if reserved else 0.0,
    }


def kv_budget(stats, total_bytes, safety):
    free = total_bytes - stats.get("allocated_bytes.all.peak", 0)
    return max(0, math.floor(free * (1.0 - safety)))


def measure_peak(fn, mem):
    mem.reset_peak_memory_stats()
    before = mem.memory_allocated()
    fn()
    return mem.max_memory_allocated() - before
```

`-(-nbytes // 512) * 512` performs integer ceiling division and maps `0` to `0`, handling empty tensors automatically. `math.prod(())` evaluates to `1`, handling scalar shapes automatically.

## Common mistakes

- **Skipping rounding or rounding only the total.** Both undercount by up to 511 bytes per tensor, which adds up rapidly with many small tensors.
- **Charging 512 bytes for an empty tensor.** A tensor with 0 elements allocates 0 bytes.
- **Assuming 4 bytes per element** instead of using `dtype.itemsize`.
- **Subtracting reserved memory in `kv_budget`.** Cached memory is reusable by the same allocator, and reserved memory stays inflated after one-time load spikes.
- **Subtracting current instead of peak allocated memory.** This forgets the activation memory required while serving a request.
- **Taking the safety margin from `total_bytes`.** With 9 of 24 GiB used, an eighth of the total is 3 GiB (a fifth of free memory). The two formulas diverge as memory fills up, and subtracting from total can go negative while free memory remains.
- **Returning a negative budget** when `peak > total_bytes`.
- **Reporting all cached bytes as releasable.** Split free blocks remain held after `empty_cache()`, making it look like memory leaked.
- **Reading a single allocator pool.** Stats exist for `small_pool` (requests under 1 MiB), `large_pool` (1 MiB and above), and their combined total `all`.
- **Indexing `stats` directly with `[]`.** `torch.cuda.memory_stats()` returns `{}` when CUDA is unavailable. PyTorch's own `memory_allocated()` uses `.get(key, 0)`.
- **Measuring the wrong counter in `measure_peak`.** Forgetting `reset_peak_memory_stats()`, subtracting allocated-before from allocated-after, returning the absolute peak without subtracting the baseline, or reading reserved memory instead of allocated memory all measure the wrong quantity.

## Complexity

`predicted_allocated` runs in $O(k)$ time for $k$ shapes. `fragmentation` and `kv_budget` run in $O(1)$ time. `measure_peak` runs in $O(1)$ time plus one call to `fn`.

## What was verified, and what was not

Verified on the CPU build of PyTorch 2.15: the key names from `torch.cuda.memory_stats` and `torch/cuda/memory.py`, the empty-dict behavior without CUDA, and the allocator constants in `c10/core/AllocatorConfig.h` (512-byte minimum block rounding, 2 MiB segments for small allocations up to 1 MiB, and 2 MiB-aligned segments for requests of 10 MiB or more).

Not verified because it requires a GPU: live `memory_stats()` outputs. The test dicts were constructed by hand, and `releasable = reserved - allocated - inactive_split` is derived from the documented definitions rather than measured directly. The 512-byte rule is also a lower bound in one case: when the allocator reuses a slightly larger cached block without splitting it, `memory_allocated()` increases by the full block size. That is why the GPU test uses tensors under 1 MiB with a 2 percent tolerance. Settings in `PYTORCH_CUDA_ALLOC_CONF` (`roundup_power2_divisions`, `max_split_size_mb`) alter rounding and splitting behavior, and the `cudaMallocAsync` backend reports some counters as zero.

## In the wild

vLLM sizes its KV cache at startup by running a dummy forward pass, measuring the peak memory usage, and assigning the cache whatever remains of `gpu_memory_utilization` times total device memory. Its margin is a fraction of total memory rather than free memory, but the subtracted baseline is still a measured peak. When PyTorch raises a CUDA out-of-memory error, the message prints both allocated memory and "reserved but unallocated" memory, suggesting allocator tuning flags when the second number is large. `torch.cuda.memory_summary()` displays the complete breakdown.
