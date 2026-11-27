Implement four functions that interpret PyTorch's CUDA caching allocator statistics: predicting allocated bytes for a list of tensor shapes, measuring memory fragmentation, computing a KV cache memory budget, and measuring a function's peak memory usage.

## Background

PyTorch does not ask the GPU driver for memory one tensor at a time. Instead, its caching allocator **reserves** large memory segments from the driver and carves out smaller blocks for individual tensors. Memory currently held by live tensors is **allocated**. When a tensor is freed, its block remains reserved in the allocator's cache so future allocations can reuse it without calling the driver.

![Reserved memory is the allocated part plus cache. `empty_cache()` returns whole free segments, but not a free block that shares its segment with a tensor.](assets/allocated-reserved.svg)

## Specification

```python
def predicted_allocated(shapes, dtype): ...  # int
def fragmentation(stats): ...  # dict
def kv_budget(stats, total_bytes, safety): ...  # int
def measure_peak(fn, mem): ...  # int
```

**`predicted_allocated(shapes, dtype)`** predicts the total allocated bytes the CUDA allocator reports for a list of tensors of the given `dtype`. `shapes` is a list of shape tuples containing non-negative integers. The empty tuple `()` represents a scalar with 1 element, and any shape containing `0` has 0 elements.

A tensor with $n$ elements holds $n \cdot \text{itemsize}$ raw bytes, where $\text{itemsize}$ is `dtype.itemsize`. Because the allocator hands out memory in 512-byte blocks, each non-empty tensor occupies

$$
512 \cdot \left\lceil \frac{n \cdot \text{itemsize}}{512} \right\rceil \text{ bytes},
$$

while a tensor with 0 elements occupies 0 bytes. Return the sum across all shapes as a Python `int` (`0` for an empty list).

**`fragmentation(stats)`** summarizes a dictionary matching the format of `torch.cuda.memory_stats()`. Missing keys default to `0` (on machines without CUDA, `memory_stats()` returns an empty dict). Read these three keys:

| key | meaning |
|---|---|
| `"allocated_bytes.all.current"` | bytes currently occupied by live tensors, $A$ |
| `"reserved_bytes.all.current"` | bytes currently held by the allocator, $R$ |
| `"inactive_split_bytes.all.current"` | free bytes inside segments that also hold a live tensor, $S$ |

Return a dictionary with these exact keys:

| key | value | type |
|---|---|---|
| `"cached_bytes"` | $R - A$, bytes held by the allocator and unused by tensors | `int` |
| `"releasable_bytes"` | $R - A - S$, bytes that `empty_cache()` can return to the device | `int` |
| `"non_releasable_fraction"` | $S / R$, or `0.0` when $R = 0$ | `float` |

**`kv_budget(stats, total_bytes, safety)`** computes how many bytes remain available for a KV cache.

- `stats` is a `memory_stats()` dictionary recorded after loading model weights and running one warm-up request.
- `total_bytes` is the total GPU memory available to this process.
- `safety` ($0 \le \text{safety} < 1$) is the fraction of *free* memory to leave unassigned.

Let $P$ be `stats["allocated_bytes.all.peak"]` (`0` if missing), the peak tensor memory observed so far. Subtract $P$ from `total_bytes` to find free memory, set aside the `safety` fraction, round down to an integer, and clamp at 0:

$$
\text{free} = \text{total\_bytes} - P,
\qquad
\text{budget} = \max\bigl(0,\ \lfloor \text{free} \cdot (1 - \text{safety}) \rfloor\bigr).
$$

Return the budget as a Python `int`.

**`measure_peak(fn, mem)`** measures how many additional bytes of tensor memory `fn` requires at its peak above what was already allocated before `fn` ran. `mem` is an object exposing three `torch.cuda` methods (on a GPU, you pass `torch.cuda` itself):

- `mem.memory_allocated()` returns the bytes currently occupied by tensors.
- `mem.max_memory_allocated()` returns the highest value `memory_allocated()` has reached since peak statistics were last reset (or since process startup).
- `mem.reset_peak_memory_stats()` resets that peak counter to the current allocated value.

Call `fn()` exactly once, discard its return value, and return the peak allocated bytes during the call minus the allocated bytes just before the call as a Python `int`. Peaks from before `measure_peak` must not affect the result, while temporary memory allocated and freed inside `fn()` must be included.

None of the four functions may modify their input arguments.

## Example

```python
predicted_allocated([(1,), (1000,), (2, 3, 4096)], torch.float16)  # 51712
```

These three `float16` tensors contain 2, 2000, and 49152 raw bytes (51154 bytes total). Rounding each up to a multiple of 512 bytes gives 512, 2048, and 49152 bytes (since $49152 = 96 \cdot 512$), for a total of 51712 bytes.

Now consider a 24 GiB GPU ($1\text{ GiB} = 2^{30}$ bytes) after loading model weights and serving one warm-up request:

```python
stats = {
    "allocated_bytes.all.current": 7516192768,  # 7 GiB: weights
    "allocated_bytes.all.peak": 9663676416,  # 9 GiB: weights + activations
    "reserved_bytes.all.current": 10737418240,  # 10 GiB
    "inactive_split_bytes.all.current": 536870912,  # 0.5 GiB
    ...  # roughly a hundred other keys
}

fragmentation(stats)
# {"cached_bytes": 3221225472,         # 3 GiB held and unused
#  "releasable_bytes": 2684354560,     # 2.5 GiB in whole free segments
#  "non_releasable_fraction": 0.05}    # 0.5 GiB of 10 GiB reserved

kv_budget(stats, 24 * 2**30, 0.125)  # 14092861440
```

Peak tensor memory is 9 GiB, leaving 15 GiB free. Keeping seven eighths of that yields 13.125 GiB (`14092861440` bytes). At 81,920 bytes per token for Granite 4.2 3B in `bfloat16` (see `kv-cache-bytes`), that fits 172,032 cached tokens. Subtracting the 10 GiB of reserved memory instead would give 12.25 GiB, and taking the safety margin from the total memory ($24 \cdot 0.875 - 9$) would give 12 GiB.

For `measure_peak`, suppose 100 MiB are currently allocated and the historical peak since process startup is 400 MiB. Inside `fn`, the code allocates a 64 MiB temporary tensor, allocates a 32 MiB result tensor, and then frees the 64 MiB temporary:

```python
measure_peak(fn, mem)  # 100663296 = 96 MiB
```

Allocated memory peaks at 196 MiB during `fn`, which is 96 MiB above the 100 MiB baseline. Without resetting peak stats first, the stale 400 MiB peak would give 300 MiB. Comparing allocated memory before and after the call would miss the temporary and give only 32 MiB.

## Constraints

- `shapes` contains up to 100 shapes with up to 5 dimensions each. All byte counts are below $2^{50}$.
- In all tests, `safety` is a multiple of $1/64$, so `free * (1 - safety)` is exact in floating point.
- All integer outputs are compared exactly and must be Python `int`s. `non_releasable_fraction` is compared with a relative tolerance of `1e-9`.
- **Execution environment.** Standard test cases run on the CPU. They check `predicted_allocated` against the 512-byte rounding rule, `fragmentation` and `kv_budget` against hand-constructed dictionaries using documented `torch.cuda.memory_stats()` keys, and `measure_peak` against a simulated allocator. Three additional cases run only on machines with an NVIDIA GPU. They verify that `predicted_allocated` is within 2 percent of `torch.cuda.memory_allocated()` for real tensors under 1 MiB, that `measure_peak(fn, torch.cuda)` detects a 64 MiB temporary, and that `fragmentation` and `kv_budget` agree with `torch.cuda`'s own counters. The GPU cases were written from the PyTorch 2.15 documentation and could not be executed when the problem was authored.

## Why this matters

An inference server decides at startup how much GPU memory to give its KV cache, and that budget sets its concurrency limit. Reading the wrong allocator counter leads directly to wasted capacity or crashes.

`nvidia-smi` reports reserved memory (plus the CUDA context), which makes a process look fuller than it is and tempts engineers into calling `empty_cache()` in a loop. Reading `memory_allocated()` while the server is idle captures model weights but misses request activations. Reading a peak counter that was never reset reflects the highest spike since startup, often from model loading. Sizing the KV cache from any of those numbers either wastes gigabytes of VRAM or triggers an out-of-memory error on the first long prompt under load.
