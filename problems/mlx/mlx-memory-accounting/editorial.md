## The idea

MLX exposes three process-wide counters for its memory allocator:

- **active**: bytes in buffers belonging to live arrays, such as model weights, caches, and results you still reference.
- **cache**: bytes in freed buffers that MLX retains so future allocations of similar size can skip the OS allocator.
- **peak**: the largest value active memory has reached since the last call to `mx.reset_peak_memory()`.

What the OS reports (such as the process footprint in Activity Monitor) is roughly `active + cache`. Because none of these counters are scoped to a single function, turning them into a per-call measurement requires a five-step protocol:

1. **Settle the inputs.** Call `mx.eval(args)` so any lazy argument is computed before measurement begins.
2. **Clean the slate.** Call `mx.synchronize()` so earlier GPU work finishes, `mx.clear_cache()` so the cache starts empty, and `mx.reset_peak_memory()` so the high-water mark starts fresh.
3. **Record the baseline.** Call `mx.get_active_memory()` to capture everything already alive (arguments, weights, and other program state). After a reset, `mx.get_peak_memory()` starts at 0 until the next allocation and then tracks total active memory, so it needs the same baseline subtracted.
4. **Run and synchronize.** Call `fn(*args)`, evaluate the result with `mx.eval(result)`, and wait for the GPU with `mx.synchronize()`.
5. **Read and subtract.** Subtract the baseline from peak and active memory, and read the cache memory directly.

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

The table below shows measurements from this implementation on an M1 Max (MLX 0.32.4) where `x` is a 16 MiB `float32` array. The numbers repeat identically across runs.

| `fn` | peak | active | cache |
|---|---|---|---|
| `a * 2.0` | 16 MiB + 4 B | 16 MiB | 4 B |
| `(a * 2.0).sum()` | 16 MiB | 4 B | 16 MiB |
| `a * 2.0 + 1.0` | 16 MiB | 16 MiB | 8 B |
| `(mx.exp(a) * 2.0 + 1.0) - 3.0` | 16 MiB | 16 MiB | 12 B |
| `(a * 2.0)[:1]` | 16 MiB | **16 MiB** | 4 B |
| `(a * 2.0)[:1] + 0.0` | 16 MiB + 8 KiB | 8 KiB | 16 MiB |

Three allocator behaviors stand out in this table:

- **Exact sizes.** There is no hidden per-array overhead at this scale, so `predicted_bytes` accurately predicts real allocations.
- **Buffer reuse in elementwise chains.** When a temporary has no other consumer, MLX writes the next result into its buffer. Four chained operations therefore peak at one array rather than four.
- **Shared buffers for slices.** A slice references its parent buffer, so keeping one row of a 16 MiB temporary holds all 16 MiB active for as long as that slice lives. Retaining a small slice from a large intermediate (such as the last token of prefill hidden states or a window of a KV cache) can unexpectedly pin a large buffer in memory.

## Common mistakes

- **Not evaluating the result.** `fn(*args)` returns a lazy graph. Without `mx.eval`, peak, active, and cache all read 0 and the call appears free. The same oversight makes timing benchmarks report microseconds for a full forward pass.
- **Reading the counters before evaluating.** Evaluating only after reading the counters still measures an unexecuted graph.
- **Skipping `mx.synchronize()`.** Immediately after `mx.eval` returns, temporary buffers may not yet be released by the GPU. In 200 of 200 runs on an M1 Max, `(a * 2.0).sum()` reported 16 MiB active without `mx.synchronize()`, versus 4 bytes with it. A short sleep masks the bug, which makes it easy to miss.
- **Counting cached memory as active.** Active plus cache is the process footprint, not what the call left in use. A function that creates a 16 MiB temporary and returns an 8 MiB array would wrongly report 24 MiB active.
- **Not resetting the peak.** Without `mx.reset_peak_memory()`, the peak reflects the highest memory usage over the entire lifetime of the process.
- **Not clearing the cache beforehand.** Freed buffers from earlier work remain in the cache and get misattributed to `fn`.
- **Omitting the baseline.** Without subtracting the pre-call active memory, the arguments and all other live arrays inflate both `peak_bytes` and `active_bytes`.
- **Ignoring lazy arguments.** If the caller passes an unevaluated array, `mx.eval(result)` evaluates that argument inside the measured region and keeps it alive afterward, charging the caller's memory to `fn`.
- **Hardcoding 4 bytes per element.** `float16` and `bfloat16` take 2 bytes per element, while `int8` and `bool_` take 1 byte. Always use `dtype.size`.
- **Treating an empty shape as 0 elements.** A scalar has shape `()` and 1 element. The empty product is 1, not 0.

## Complexity

`predicted_bytes` runs in time linear in the total number of dimensions across `shapes`. `measure` adds two GPU synchronizations and a cache clear to the cost of `fn`. Because clearing the cache forces subsequent allocations back to the OS, avoid leaving `measure` inside a hot loop.

## In the wild

`mlx_lm` prints `mx.get_peak_memory()` after generation ("Peak memory: ... GB") without subtracting a baseline, intentionally including the model weights. On an M1 Max, MLX's default cache limit (returned as the previous value by `mx.set_cache_limit`) is 60.8 GiB, which is nearly all of RAM. Because MLX rarely returns freed buffers to the OS on its own, inference servers handling variable-shape requests periodically call `mx.clear_cache()` or configure a lower cache limit. The judge's `mlx_memory` context manager used in other problems on this track is built on the same protocol you just implemented.
