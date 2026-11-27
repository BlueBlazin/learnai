Implement `run_batched` to run a model over a tensor of items in batches and recover from out-of-memory errors. When a batch runs out of GPU memory, free what the failed attempt held, halve the batch size, and retry the same items.

## Specification

```python
def run_batched(model, items, batch_size, empty_cache, grow_after=None): ...
```

- `model(batch)` takes a tensor of `b` rows and returns a tensor of `b` rows, where each output row corresponds to the input row at the same index. It raises `torch.OutOfMemoryError` when it cannot allocate the memory needed for the batch, and it may raise other exceptions.
- `items` is a tensor with $N \ge 1$ rows along axis 0.
- `batch_size` is the initial and maximum number of rows per batch (at least 1).
- `empty_cache` is a zero-argument callable. On a GPU this is `torch.cuda.empty_cache`, passed in so the judge can inspect calls to it.
- `grow_after` is `None` or a positive integer.
- Return a single tensor formed by concatenating all batch outputs along axis 0, so row `i` of the result is the output for `items[i]`.

Maintain a starting index `start` (initially `0`) and a current batch size `size` (initially `batch_size`). Repeat until `start == N`:

1. **Try** the slice `items[start : start + size]`, which has $b = \min(\text{size}, N - \text{start})$ rows.
2. **If `model` raises `torch.OutOfMemoryError`:**
   - If $b = 1$, re-raise the error because a single item does not fit.
   - Otherwise, call `empty_cache()` once, set `size` to $\lfloor b / 2 \rfloor$ (half of the batch that failed), and retry from step 1 with the same `start`.
3. **If `model` succeeds**, save its output and advance `start` by $b$.

When `grow_after` is `None`, the reduced `size` stays in effect for all remaining batches. When `grow_after` is a positive integer $g$, count consecutive successful batches executed while `size < batch_size`. After $g$ consecutive successes, update `size` to $\min(2 \cdot \text{size}, \text{batch\_size})$ and reset the count to 0. Any `torch.OutOfMemoryError` also resets the count to 0. Batches that succeed at `batch_size` are not counted.

**The failed batch must be unreachable when `empty_cache()` runs.** The batch passed to `model` represents the memory held by the failed call. At the moment `empty_cache()` is called, nothing may reference that batch. Neither a local variable in your function nor the active exception (whose traceback keeps the failed call's stack frames and tensors alive until exception handling finishes) may still hold a reference to it. Call `empty_cache()` only after a failure that will be retried. Never call it after a successful batch or before re-raising when $b = 1$.

Any exception other than `torch.OutOfMemoryError` must propagate immediately and unchanged. Do not modify `items`.

## Shapes

| Name | Shape | Dtype |
|---|---|---|
| `items` | `(N, ...)` | any |
| `model(batch)` (for `b` rows) | `(b, ...)` | any |
| return value | `(N, ...)` with the model's trailing dimensions | same as `model` output |

## Example

Consider 10 items, `batch_size = 8`, and a model that fits at most 3 rows in memory.

```python
run_batched(model, items, 8, empty_cache)
```

| Batch tried | Rows | Outcome | Next action |
|---|---|---|---|
| `items[0:8]` | 8 | out of memory | `empty_cache()`, `size = 4` |
| `items[0:4]` | 4 | out of memory | `empty_cache()`, `size = 2` |
| `items[0:2]` | 2 | ok | advance `start` to 2 |
| `items[2:4]` | 2 | ok | advance `start` to 4 |
| `items[4:6]`, `[6:8]`, `[8:10]` | 2 | ok | advance `start` to 10 |

The model is called 7 times with batch sizes 8, 4, 2, 2, 2, 2, 2. `empty_cache` is called twice, and the output has 10 rows in the order of `items`.

With `grow_after=2`, the loop attempts to double `size` after every two consecutive successes below `batch_size`:

| Batch tried | Rows | Outcome | Next action |
|---|---|---|---|
| `items[0:8]`, `items[0:4]` | 8, 4 | out of memory | as above, `size = 2` |
| `items[0:2]`, `items[2:4]` | 2, 2 | ok, ok | 2 successes, `size = 4` |
| `items[4:8]` | 4 | out of memory | `empty_cache()`, `size = 2`, count 0 |
| `items[4:6]`, `items[6:8]` | 2, 2 | ok, ok | 2 successes, `size = 4` |
| `items[8:10]` | 2 | ok | only 2 items remain |

This run makes 8 model calls with batch sizes 8, 4, 2, 2, 4, 2, 2, 2 and calls `empty_cache` three times. Against a fixed memory limit, growing back only triggers extra failures. It pays off when the failure was transient, such as a temporary memory spike from another request.

## Constraints

- $1 \le N \le 40$, $1 \le \text{batch\_size} \le 64$, $1 \le \text{grow\_after} \le 4$.
- The output is compared against a single call of the model on all items with a relative tolerance of `1e-5` and an absolute tolerance of `1e-6`.
- The sequence of batch sizes tried is checked for an exact match. The number of `empty_cache` calls (reported as `empty_cache_calls`) must equal one per retried failure.
- The judge's model refuses to be called more than 80 times.
- **What runs where.** Cases without a hardware mark run on the CPU with a model that raises `torch.OutOfMemoryError` above a row limit (and in some cases once on a chosen call even when the batch fits). The judge logs every call and holds a weak reference to each failed batch to verify it is no longer alive when `empty_cache` runs. One case runs only on a machine with an NVIDIA GPU, using the same model and checks with `items` on CUDA and the real `torch.cuda.empty_cache`. It does not provoke a real hardware OOM, whose threshold depends on the device. The GPU case was written from the PyTorch 2.15 documentation and could not be run when the problem was authored.

## Why this matters

An inference server that batches requests cannot always predict exact memory usage in advance because it depends on prompt length, image count, or beam width. Out-of-memory errors happen in normal operation, and recovering cleanly separates a resilient server from one that crashes.

A common bug is retrying inside the `except` block. While the exception is still being handled, its traceback pins the activations of the failed attempt in memory, so the retry fails for lack of the memory it expected to reuse and crashes with a second out-of-memory error. Quieter bugs include catching `RuntimeError` (which treats shape or device bugs as OOMs and halves until failure), restarting from index 0 (producing duplicate rows), and calling `empty_cache()` after every batch (forcing every batch to pay for fresh driver allocations).
