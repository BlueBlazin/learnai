Answer four questions about GPU memory from the numbers PyTorch's CUDA allocator
reports: how much a set of tensors will occupy, how much of what the process holds is
merely cached, how many bytes are left for a KV cache, and how much a function needs
at its peak. Each answer is a line of arithmetic, and each has a number next to the
right one that is easy to take instead.

## Specification

```python
def predicted_allocated(shapes, dtype): ...  # int
def fragmentation(stats): ...  # dict
def kv_budget(stats, total_bytes, safety): ...  # int
def measure_peak(fn, mem): ...  # int
```

PyTorch does not ask the GPU driver for memory tensor by tensor. Its caching
allocator **reserves** large segments from the driver and carves the blocks that
tensors occupy out of them; memory occupied by tensors is **allocated**. When a
tensor is freed its block stays reserved, as cache, and is handed out again.

![The memory of a device: tensors occupy the allocated part, and the rest of the reserved memory is cache. `empty_cache()` can return whole free segments to the device, but not a free block that shares its segment with a tensor.](assets/allocated-reserved.svg)

**`predicted_allocated(shapes, dtype)`** predicts what the allocator will report as
allocated for a list of tensors. The allocator hands out memory in multiples of 512
bytes, so a tensor of $n$ elements of `dtype` occupies

$$
512 \cdot \left\lceil \frac{n \cdot \text{itemsize}}{512} \right\rceil \text{ bytes},
$$

and a tensor without elements occupies nothing. `shapes` is a list of shapes, each a
tuple of non-negative ints; the empty tuple `()` is a scalar with one element, and a
shape containing a `0` has no elements. `dtype` is a torch dtype. Return the sum over
all shapes as a Python `int` (`0` for an empty list).

**`fragmentation(stats)`** summarizes a dict of the form `torch.cuda.memory_stats()`
returns. It uses three of its keys, and a missing key counts as `0` (on a machine
without CUDA the dict is empty):

| key | meaning |
|---|---|
| `"allocated_bytes.all.current"` | bytes occupied by tensors now, $A$ |
| `"reserved_bytes.all.current"` | bytes the allocator holds now, $R$ |
| `"inactive_split_bytes.all.current"` | free bytes inside segments that also hold a tensor, $S$ |

It returns a dict with exactly these keys:

| key | value | type |
|---|---|---|
| `"cached_bytes"` | $R - A$: held by the allocator, occupied by no tensor | `int` |
| `"releasable_bytes"` | $R - A - S$: what `empty_cache()` can return to the device | `int` |
| `"non_releasable_fraction"` | $S / R$, and `0.0` when $R = 0$ | `float` |

**`kv_budget(stats, total_bytes, safety)`** returns the number of bytes a KV cache may
take. `stats` is a `memory_stats()` dict read after the weights were loaded and one
warm-up request was served; `total_bytes` is the memory this process may use on the
device; `safety`, $0 \le \text{safety} < 1$, is the fraction of the *free* memory to
leave unassigned. With $P$ = `stats["allocated_bytes.all.peak"]` (`0` if missing), the
high-water mark of tensor memory:

$$
\text{free} = \text{total\_bytes} - P,
\qquad
\text{budget} = \max\bigl(0,\ \lfloor \text{free} \cdot (1 - \text{safety}) \rfloor\bigr).
$$

Return a Python `int`.

**`measure_peak(fn, mem)`** returns how many bytes of tensor memory `fn` needs at its
worst moment, over and above what was allocated when it started. `mem` is an object
with the functions of `torch.cuda` of the same names (on a GPU, pass the module
itself):

- `mem.memory_allocated()`: bytes occupied by tensors now;
- `mem.max_memory_allocated()`: the largest value `memory_allocated()` has had since
  the peak statistics were last reset (or since the process started);
- `mem.reset_peak_memory_stats()`: resets that peak to the current value.

Call `fn()` exactly once (its return value is discarded) and return, as a Python
`int`, the peak of allocated memory during the call minus the allocated memory just
before it. Peaks reached before the call do not count; memory that `fn` allocates and
frees again does.

The functions do not modify their arguments.

## Example

```python
predicted_allocated([(1,), (1000,), (2, 3, 4096)], torch.float16)  # 51712
```

The three float16 tensors hold 2, 2000, and 49152 bytes. They occupy 512, 2048, and
49152 bytes (the last is already $96 \cdot 512$): 51712 in total, not the 51154 of
their data.

A 24 GiB GPU, after loading a model and serving one request (1 GiB = $2^{30}$ bytes):

```python
stats = {
    "allocated_bytes.all.current": 7516192768,  # 7 GiB: the weights
    "allocated_bytes.all.peak": 9663676416,  # 9 GiB: weights and activations
    "reserved_bytes.all.current": 10737418240,  # 10 GiB
    "inactive_split_bytes.all.current": 536870912,  # 0.5 GiB
    ...  # about a hundred other keys
}

fragmentation(stats)
# {"cached_bytes": 3221225472,         3 GiB held and unused
#  "releasable_bytes": 2684354560,     2.5 GiB of it in whole free segments
#  "non_releasable_fraction": 0.05}    0.5 GiB of the 10 GiB reserved

kv_budget(stats, 24 * 2**30, 0.125)  # 14092861440
```

The process needs 9 GiB at its peak, which leaves 15 GiB; seven eighths of that is
13.125 GiB. At the 81,920 bytes per token of Granite 4.2 3B in bfloat16
(`kv-cache-bytes`) that is 172,032 cached tokens. Subtracting the 10 GiB that are
reserved instead would give 12.25 GiB, and taking the margin from the total,
$24 \cdot 0.875 - 9$, would give 12 GiB.

For `measure_peak`, suppose 100 MiB are allocated, the peak since the process started
is 400 MiB, and `fn` allocates a 64 MiB temporary, then a 32 MiB result, then frees
the temporary:

```python
measure_peak(fn, mem)  # 100663296 = 96 MiB
```

Allocated memory rises to 196 MiB during the call. Without a reset the old peak of
400 MiB hides it (the answer would be 300 MiB); comparing the allocated memory after
the call with the memory before it gives 32 MiB.

## Constraints

- Up to 100 shapes of up to 5 dimensions; all byte counts are below $2^{50}$.
- In the tests `safety` is a multiple of $1/64$, so that `free * (1 - safety)` is exact
  in floating point however it is written.
- All integer results are compared exactly and must be Python `int`s;
  `non_releasable_fraction` is compared to a relative tolerance of `1e-9`.
- **What runs where.** The cases without a hardware mark run on the CPU:
  `predicted_allocated` against the rounding rule, `fragmentation` and `kv_budget` on
  dicts that have the keys of `torch.cuda.memory_stats()` and consistent values, and
  `measure_peak` against a simulated allocator. Those dicts were built by hand from
  the documented key names; they were not captured from a GPU. Three cases run only
  on a machine with an NVIDIA GPU: the prediction must be within 2 percent of what
  `torch.cuda.memory_allocated()` reports for real tensors (each below 1 MiB),
  `measure_peak(fn, torch.cuda)` must see a 64 MiB temporary, and `fragmentation` and
  `kv_budget` must agree with `torch.cuda`'s own counters. The GPU cases were written
  from the PyTorch 2.15 documentation and could not be run when the problem was
  authored.

## Why this matters

An inference server decides at startup how much memory its KV cache gets, and that
decision is its concurrency limit. The three wrong numbers are all on the screen at
the same time. `nvidia-smi` shows reserved memory (plus the CUDA context), which
makes a process look fuller than it is and tempts people into calling
`empty_cache()` in a loop. `memory_allocated()` read while the server is idle shows
the weights and none of the activations a request needs. And a peak that was never
reset reports the largest allocation since the process started, usually from model
loading. Sizing the cache from any of them either wastes gigabytes or produces an
out-of-memory error on the first long prompt under load.
