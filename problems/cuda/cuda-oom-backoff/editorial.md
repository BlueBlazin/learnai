## The idea

`torch.OutOfMemoryError` is raised from deep inside the model, in the middle of a
forward pass, with activations of the oversized batch alive in the frames of every
layer that was running. Python keeps those frames after the error: an exception
carries its traceback, and a traceback is a chain of references to the frames it
passed through, local variables included. As long as the exception is being handled,
that is, until the `except` block ends, all of that memory is still occupied by
tensors. PyTorch's FAQ ("My out of memory exception handler can't allocate memory")
describes exactly this and gives the fix used here: note the failure inside the
handler, recover outside it.

`torch.cuda.empty_cache()` has a narrow meaning. It returns to the driver the blocks
that the caching allocator holds and no tensor occupies; it cannot free a tensor.
Called inside the handler, it finds the failed attempt's blocks occupied and returns
nothing. Called after the handler, with no reference left, it returns the large
blocks of the attempt, so that the smaller retry is not served from a pool
fragmented by them. It is not a way to make more memory available to PyTorch in
general (the documentation says so explicitly, and the allocator already flushes its
cache and retries once before it raises), which is why calling it after every batch
only costs time: the next batch has to obtain its blocks from the driver again.

The rest is a retry policy with exact rules:

- **Halve what failed**, not the setting. A last slice of 3 items that fails at
  `size = 8` should be retried with 1 item, not with 4 (which is the same 3).
- **Retry in place.** `start` moves only on success, so no row is produced twice and
  the outputs stay in order.
- **One row is the floor.** If a single item does not fit, nothing this function can
  do will help, and the caller must see the error.
- **Remember.** Without `grow_after`, the reduced size is kept: the limit that was
  just discovered is a property of the model and the device, and rediscovering it on
  every slice costs a failed forward pass each time.
- **Or probe.** With `grow_after`, the size doubles after a run of successes, and the
  run restarts at every failure. That recovers from a transient failure in a
  long-running server.

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

The slice `items[start:end]` is an argument expression, not a variable, so when the
call raises, the function itself holds no reference to the batch. `except
torch.OutOfMemoryError:` without `as exc` binds nothing; with `as exc`, Python
deletes the name at the end of the block, which works as well. A bare `raise` inside
the handler re-raises the same exception object with its original traceback.

## Common mistakes

- **Recovering inside the `except` block.** `empty_cache()` there releases nothing,
  and a retry there runs with the failed attempt still in memory.
- **A variable that names the batch** (`batch = items[start:end]` before the `try`)
  and is still bound when `empty_cache()` runs. Use the expression directly, or
  `del batch` first. In a real pipeline the batch is typically a fresh device tensor
  (`batch.to(device)`), and it is exactly the memory that must go.
- **No `empty_cache()` between the failure and the retry**, and the opposite,
  **`empty_cache()` after every batch**.
- **`except RuntimeError`.** `OutOfMemoryError` is a subclass of `RuntimeError`, and
  so are shape mismatches, device mismatches, and CUDA device-side assertions. They
  are bugs, they fail at every batch size, and the loop answers them by halving.
- **Retrying at the same size.** Nothing changed, so nothing changes.
- **Halving past one.** `1 // 2` is 0, the empty batch succeeds, and the position
  never advances.
- **Restarting at the first item** after a failure while keeping the outputs so far.
- **Forgetting the reduced size**: resetting to `batch_size` for the next slice, so
  that every slice fails first.
- **Ignoring `grow_after`**, **growing past `batch_size`**, and **not resetting the
  success count at a failure**.
- **Halving the size setting** instead of the batch that failed.
- **Putting the failed slice aside** and retrying it after the other items. Every
  row is computed once, and the result is in the wrong order.

## Complexity

With a fixed limit of $m$ rows and no growth, there are at most
$\lceil \log_2(\text{batch\_size} / m) \rceil$ failed calls, all on the first slice,
and about $N / m'$ successful ones, where $m'$ is the size the halving arrives at
(between $m / 2$ and $m$). With `grow_after = g` against a fixed limit, one in every
$g + 1$ calls is a failed probe.

## In the wild

Hugging Face Accelerate's `find_executable_batch_size` decorator reruns a function
with a smaller batch size after an out-of-memory error, collecting garbage and
emptying the cache between attempts, and never grows back. Serving engines mostly avoid the situation by reserving
memory up front (a KV cache sized at startup, a cap on batched tokens), and keep a
handler of this kind as the last line of defense. The CUDA notes also describe
allocator options for the same purpose, such as a per-process memory fraction that
turns "the device is full" into a catchable `OutOfMemoryError` earlier.

What was verified where: `torch.OutOfMemoryError` exists in torch 2.15, is a
subclass of `RuntimeError`, and is the same class as `torch.cuda.OutOfMemoryError`;
the lifetime of a failed batch inside and outside the handler was observed on the
CPU with weak references. That a real allocation failure raises this class and that
`empty_cache()` then releases the attempt's blocks is taken from the documentation
(the FAQ and the CUDA notes); the GPU case has not been run.
