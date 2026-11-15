## The idea

Asking the GPU driver for memory is slow, so PyTorch asks rarely. Its caching
allocator reserves segments from the driver, cuts blocks out of them for tensors,
and keeps a freed block for the next request instead of giving it back. That one
design decision produces three different numbers for "how much memory is used":

- **allocated**: the blocks that tensors occupy now (`torch.cuda.memory_allocated()`);
- **reserved**: every segment the allocator holds, occupied or not
  (`torch.cuda.memory_reserved()`, and roughly what `nvidia-smi` shows for the
  process, which also includes the CUDA context);
- **peak allocated**: the largest value allocated has had since the peak statistics
  were last reset (`torch.cuda.max_memory_allocated()`).

In torch 2.15 the three functions are one-line lookups in the dict that
`torch.cuda.memory_stats()` returns: `allocated_bytes.all.current`,
`reserved_bytes.all.current`, and `allocated_bytes.all.peak`.

Each question of the problem is about one of them.

*How much will these tensors occupy?* Allocated, and the allocator's unit is 512
bytes: the CUDA notes say sizes "are rounded up in multiple of blocks size of 512".
A one-element tensor occupies 512 bytes; a hundred of them occupy 51200. Rounding
the sum instead of each term is the natural mistake.

*How much of what the process holds is just cache?* Reserved minus allocated. Not all
of it can be returned to the device: `empty_cache()` releases free segments, and a
free block that shares its segment with a live tensor is stuck there. The allocator
counts those bytes as `inactive_split_bytes`, documented as "inactive, non-releasable
memory" (`torch.cuda.memory_summary()` prints the same number as "Non-releasable
memory"). A large non-releasable share is what fragmentation looks like in these
statistics.

*How much is left for a cache?* The cache will be allocated through the same
allocator, which satisfies requests from its own free blocks before reserving more.
So cached memory is available, and what is unavailable is what tensors occupy at the
worst moment of a request: the peak of allocated. The safety margin is a fraction of
the free memory because it stands for what the measurement cannot see: fragmentation
and allocations larger than those of the warm-up.

*How much does this function need?* Peak allocated during the call, minus allocated
before it. The peak counter never decreases on its own, so it has to be reset first;
and the difference of allocated before and after says nothing about temporaries.

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

`-(-n // 512) * 512` is integer ceiling division; it maps 0 to 0, which is the
empty-tensor rule for free. `math.prod(())` is 1, which is the scalar rule.

## Common mistakes

- **No rounding**, or **rounding the total once**. Both undercount, by up to 511
  bytes per tensor; with many small tensors that is most of the memory.
- **A block for an empty tensor.** A tensor with no elements allocates nothing.
- **Four bytes per element** for every dtype.
- **Reserved treated as in use** in the budget. The cache then never gets the memory
  the allocator is holding for exactly such a request, and the number depends on the
  history of the process (a large temporary at load time inflates reserved for good).
- **Current instead of peak allocated.** The budget includes the memory that the
  next request's activations need.
- **The safety margin taken from the total.** With 9 of 24 GiB in use, an eighth of
  the total is 3 GiB, a fifth of what is free; the two definitions diverge as the
  model fills the device, and this one goes negative while memory is still free.
- **A negative budget** when the peak exceeds the total.
- **All cached memory reported as releasable.** `empty_cache()` then appears to
  "leak": the split blocks stay reserved.
- **Reading one pool.** The statistics exist per pool (`small_pool` for requests
  below 1 MiB, `large_pool` for the rest) and summed (`all`).
- **Indexing the dict.** `torch.cuda.memory_stats()` is empty on a machine without
  CUDA; `torch.cuda.memory_allocated()` itself uses `.get(key, 0)`.
- **No reset before measuring a peak**, **allocated-after minus allocated-before**,
  **the absolute peak**, and **reserved instead of allocated**: the four ways to
  measure something other than what the function needed.

## Complexity

Linear in the number of shapes; constant for the other three, plus one call of `fn`.

## What was verified, and what was not

Verified on the CPU build of torch 2.15: the key names (from the docstring of
`torch.cuda.memory_stats` and from `torch/cuda/memory.py`, where the four functions
above are lookups of those keys), the empty dict without CUDA, and the allocator's
constants in the header that ships with the wheel (`c10/core/AllocatorConfig.h`:
"all sizes are rounded to at least 512 bytes", small allocations are those up to
1 MiB and are packed into 2 MiB segments, requests of 10 MiB and more reserve
segments rounded up to 2 MiB).

Not verified, because it needs a GPU: the values of a real `memory_stats()` dict. The
dicts in the tests were built by hand, and the identity
`releasable = reserved - allocated - inactive_split` is derived from the documented
meaning of the three statistics, not measured. The 512-byte rule is a lower bound in
one respect: when the allocator reuses a cached block that is only slightly larger
than the request, it may hand out the whole block without splitting it, and
allocated then grows by the block's size. That is why the GPU case uses tensors
below 1 MiB and allows 2 percent. The environment variable `PYTORCH_CUDA_ALLOC_CONF`
(`roundup_power2_divisions`, `max_split_size_mb`) changes the rounding and splitting
rules, and the `cudaMallocAsync` backend reports some statistics as zero.

## In the wild

vLLM sizes its KV cache at startup from a profiling run: it executes a dummy forward
pass, measures the peak, and gives the cache what remains of
`gpu_memory_utilization` times the device's memory. Its margin is a fraction of the
total, a different convention from the one specified here; whichever is used, the
quantity subtracted is a measured peak. The out-of-memory message of
PyTorch itself prints allocated and "reserved but unallocated" memory and suggests
allocator settings when the second number is large, which is the fragmentation
figure of this problem. `torch.cuda.memory_summary()` prints the full table.
