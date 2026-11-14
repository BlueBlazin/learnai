## The idea

MLX keeps three numbers about its allocator, for the whole process:

- **active**: bytes in buffers that belong to arrays which are alive (weights, a
  cache, results you still hold);
- **cache**: bytes in buffers that were freed and are kept, so the next allocation of
  that size does not have to ask the system;
- **peak**: the largest value *active* has had since the last
  `mx.reset_peak_memory()`.

What the system sees (the process's footprint in Activity Monitor) is roughly active
plus cache. None of the three knows anything about "this function call". A
measurement turns global counters into a per-call figure with a protocol:

1. **Settle the inputs.** `mx.eval(args)`: a lazy argument is work that has not
   happened yet, and it would happen inside the measured region.
2. **Clean the slate.** `mx.synchronize()` so nothing earlier is still running;
   `mx.clear_cache()` so that what is in the cache afterwards was put there by this
   call; `mx.reset_peak_memory()` so that the high-water mark starts over.
3. **Take a baseline.** `mx.get_active_memory()` now is everything that is not this
   call's: the arguments, the model, whatever else the program holds. (After a reset
   the peak reads 0 until the next allocation and then tracks total active memory, so
   it needs the same baseline subtracted.)
4. **Do the work, for real.** Call `fn`, `mx.eval` its result, `mx.synchronize()`.
5. **Read and subtract.**

## Reference solution (MLX)

```python
def predicted_bytes(shapes, dtype):
    return sum(math.prod(shape) for shape in shapes) * dtype.size


def measure(fn, *args):
    mx.eval(args)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = mx.get_active_memory()

    result = fn(*args)
    mx.eval(result)
    mx.synchronize()

    return {
        "result": result,
        "peak_bytes": mx.get_peak_memory() - baseline,
        "active_bytes": mx.get_active_memory() - baseline,
        "cache_bytes": mx.get_cache_memory(),
    }
```

## What the numbers look like

All measured with this function on this machine (M1 Max, MLX 0.32.4), with `x` a
16 MiB float32 array. The figures repeat exactly from run to run.

| `fn` | peak | active | cache |
|---|---|---|---|
| `a * 2.0` | 16 MiB + 4 B | 16 MiB | 4 B |
| `(a * 2.0).sum()` | 16 MiB | 4 B | 16 MiB |
| `a * 2.0 + 1.0` | 16 MiB | 16 MiB | 8 B |
| `(mx.exp(a) * 2.0 + 1.0) - 3.0` | 16 MiB | 16 MiB | 12 B |
| `(a * 2.0)[:1]` | 16 MiB | **16 MiB** | 4 B |
| `(a * 2.0)[:1] + 0.0` | 16 MiB + 8 KiB | 8 KiB | 16 MiB |

Three things worth knowing are visible here. Sizes are exact: there is no hidden
per-array overhead at this scale, so `predicted_bytes` is a prediction you can hold
MLX to. A chain of elementwise operations does not need a buffer per operation: when
a temporary has no other user, MLX computes the next result into its buffer, so four
operations peak at one array, not four. And a slice shares its parent's buffer: one
row of a 16 MiB temporary keeps all 16 MiB active for as long as the row is alive.
That last one is how a "small" tensor kept from a large intermediate (the last
position of a prefill's hidden states, a window of a KV cache) holds on to far more
than its size.

## Common mistakes

- **Not evaluating the result.** `fn(*args)` returns a graph. Peak, active, and cache
  are all 0, and the measured function looks free. The same mistake makes a benchmark
  report microseconds for a forward pass.
- **Reading the counters before evaluating**, then evaluating so that the result is
  usable. The report is still about a graph that had not run.
- **Not waiting for the GPU.** Immediately after `mx.eval` returns, the temporaries of
  the graph may not have been released yet: in 200 of 200 runs on this machine,
  `(a * 2.0).sum()` read as 16 MiB still active without `mx.synchronize()`, and as 4
  bytes with it. A short sleep "fixes" it too, which is how this bug hides.
- **Counting the cache as active.** Active plus cache is what the process holds, not
  what the call left in use. A function that makes a 16 MiB temporary and returns 8
  MiB would be reported as leaving 24 MiB.
- **Not resetting the peak.** The peak is then the high-water mark of everything the
  process ever did: loading the model, an earlier and larger experiment.
- **Not clearing the cache first.** Buffers freed by earlier work are reported as if
  this call had left them.
- **No baseline.** Without subtracting the active memory from before the call, the
  arguments and everything else alive are added to both peak and active.
- **Lazy arguments.** If the caller passes a graph, evaluating `fn`'s result also
  evaluates the argument, inside the measured region, and the argument stays alive
  afterwards: 16 MiB of the caller's memory charged to an 8 MiB function.
- **Four bytes per element.** Half-precision weights, int8 codes, and boolean masks
  have 2, 1, and 1; use the dtype's own size.
- **A scalar has one element.** The product over an empty shape is 1, not 0.

## Complexity

`predicted_bytes` is linear in the number of shapes. `measure` adds two
synchronizations and a cache clear to the cost of `fn`; clearing the cache means the
next allocations go to the system again, so do not leave a measurement inside a hot
loop.

## In the wild

`mlx_lm` prints `mx.get_peak_memory()` after a generation ("Peak memory: ... GB"),
which is this measurement without the baseline: the model's weights are included, on
purpose. On this machine the default cache limit is 60.8 GiB (what
`mx.set_cache_limit` returns as the previous value), close to all of RAM, so MLX
practically never returns a freed buffer to the system by itself; servers that run
many differently shaped requests call `mx.clear_cache()` between them or set a lower
limit. The kit's `mlx_memory` context manager, which the judge uses in other
problems of this track, is the function you just wrote.
