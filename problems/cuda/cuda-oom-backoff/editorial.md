## The idea

When `torch.OutOfMemoryError` is raised inside a forward pass, intermediate activations from the oversized batch are still alive in the stack frames of the active layers. Python preserves those frames after the exception is raised because an exception carries its traceback, and a traceback holds references to every frame it passed through along with their local variables. Until the `except` block exits, all of those tensors stay allocated. PyTorch's FAQ ("My out of memory exception handler can't allocate memory") explains this mechanism and gives the standard fix: record the failure inside the `except` block, then recover after the `except` block ends.

`torch.cuda.empty_cache()` only returns unused blocks held by PyTorch's caching allocator back to the CUDA driver. It cannot free live tensors. If you call it inside the `except` block, the failed attempt's blocks are still occupied and nothing is released. Calling it after the `except` block, once no references remain, releases those large blocks so the smaller retry does not run in a fragmented pool. Calling `empty_cache()` after every successful batch only hurts performance because PyTorch's allocator already flushes its cache and retries once before raising an OOM, so extra calls just force the next batch to request fresh memory from the driver.

The retry policy follows five rules:

- **Halve the batch that failed, not `size`.** If a final slice of 3 items fails when `size = 8`, the next attempt should use $\lfloor 3 / 2 \rfloor = 1$ item, not 4 (which would retry the exact same 3 items).
- **Retry in place.** Advance `start` only after a batch succeeds so every row is computed once and outputs stay in order.
- **Stop at one row.** If a single item does not fit, halving cannot help, so re-raise the error.
- **Keep the reduced size when `grow_after` is `None`.** The limit discovered by halving reflects the model and device, so resetting to `batch_size` on every slice would waste a failed forward pass each time.
- **Probe upward when `grow_after` is set.** Doubling `size` after a streak of successes, and resetting the streak on any OOM, lets a long-running server recover after a transient memory spike.

## Reference solution (PyTorch)

```python
def run_batched(model, items, batch_size, empty_cache, grow_after=None):
    outputs = []
    n = len(items)
    start, size, streak = 0, batch_size, 0
    while start < n:
        end = min(start + size, n)
        oom = False
        try:
            outputs.append(model(items[start:end]))
        except torch.OutOfMemoryError:
            if end - start == 1:
                raise
            oom = True

        if oom:
            empty_cache()
            size = (end - start) // 2
            streak = 0
            continue

        start = end
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size = min(2 * size, batch_size)
                streak = 0
    return torch.cat(outputs, dim=0)
```

Passing `items[start:end]` directly as an argument avoids binding the slice to a local variable, so the function holds no reference to the failed batch once the call raises. Using `except torch.OutOfMemoryError:` without `as exc` binds no exception variable (Python also clears an `as exc` target when the block exits). A bare `raise` re-raises the original exception with its original traceback.

## Common mistakes

- **Recovering inside the `except` block.** Calling `empty_cache()` there releases nothing, and retrying there runs while the failed attempt is still in memory.
- **Binding the batch to a local variable.** Writing `batch = items[start:end]` before `try` keeps `batch` alive when `empty_cache()` runs unless you `del batch` first. In real pipelines the batch is often a fresh GPU copy (`batch.to(device)`), which must be freed before retrying.
- **Calling `empty_cache()` at the wrong times.** Skipping `empty_cache()` before retrying leaves fragmented blocks, while calling it after every batch wastes time on driver allocations.
- **Catching `RuntimeError`.** `OutOfMemoryError` inherits from `RuntimeError`, as do shape mismatches, device mismatches, and CUDA device-side assertions. Catching `RuntimeError` treats real bugs as OOMs and repeatedly halves the batch.
- **Retrying at the same batch size.** Running the exact same slice again fails the same way.
- **Halving past 1.** Computing `1 // 2` gives `0`, so an empty slice succeeds without advancing `start` and loops forever.
- **Restarting at index 0 after a failure.** Keeping earlier outputs while resetting `start = 0` duplicates rows.
- **Resetting `size` to `batch_size` on the next slice.** Every subsequent slice pays for a failed attempt before shrinking again.
- **Broken `grow_after` tracking.** Ignoring `grow_after`, allowing `size` to exceed `batch_size`, or forgetting to reset the success count on an OOM all produce the wrong batch sequence.
- **Halving `size` instead of the failed batch size.** On a short trailing slice, `size // 2` can still be larger than or equal to the slice that just failed.
- **Deferring the failed slice to the end.** Processing later items first computes every row once, but returns rows in the wrong order.

## Complexity

Let $m$ be a fixed maximum number of rows that fit in memory, and let $m'$ be the batch size that halving settles on (between $m / 2$ and $m$). Without `grow_after`, there are at most $\lceil \log_2(\text{batch\_size} / m) \rceil$ failed calls, all on the first slice, and about $N / m'$ successful calls. With `grow_after = g` against a fixed limit, one in every $g + 1$ calls is a failed probe.

## In the wild

Hugging Face Accelerate provides the `find_executable_batch_size` decorator, which catches out-of-memory errors, halves the batch size, runs garbage collection and `empty_cache()`, and retries without growing back. Production serving engines usually prevent OOMs up front by preallocating a fixed KV cache and capping batched tokens, while keeping an OOM backoff handler as a fallback. PyTorch's CUDA allocator also exposes a per-process memory fraction setting to trigger a catchable `OutOfMemoryError` before the device runs completely dry.

Verification note: on PyTorch 2.15, `torch.OutOfMemoryError` is a subclass of `RuntimeError` and aliases `torch.cuda.OutOfMemoryError`. The failed batch lifetime inside and outside the `except` block was verified on CPU with weak references. Real CUDA allocation failures and `empty_cache()` block release follow the PyTorch FAQ and CUDA documentation. The GPU case has not been run on hardware.
