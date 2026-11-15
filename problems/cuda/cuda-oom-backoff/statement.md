Run a model over a list of items in batches, and when a batch runs out of GPU memory,
recover: free what the failed attempt held, halve the batch, and retry the same items.
The retry policy is a few lines; the part that goes wrong is *when* the memory of the
failed attempt can be freed at all.

## Specification

```python
def run_batched(model, items, batch_size, empty_cache, grow_after=None): ...
```

- `model(batch)` maps a tensor of `b` rows to a tensor of `b` rows, each output row
  computed from the input row at the same index. It raises `torch.OutOfMemoryError`
  when it cannot allocate what the batch needs, and it may raise other exceptions.
- `items`: a tensor of shape `(N, ...)`, $N \ge 1$.
- `batch_size`: the initial, and the largest, number of rows per batch, at least 1.
- `empty_cache`: a function of no arguments. On a GPU it is `torch.cuda.empty_cache`;
  it is passed in so that the judge can watch it.
- `grow_after`: `None`, or a positive int.
- Returns one tensor: the outputs of all batches concatenated along axis 0, so that
  row `i` of the result is the output for `items[i]`.

The function keeps a position `start` (initially 0) and a current size `size`
(initially `batch_size`) and repeats, until `start` reaches `N`:

1. **Try** the batch `items[start : start + size]`, which has
   $b = \min(\text{size}, N - \text{start})$ rows.
2. **If the model raises `torch.OutOfMemoryError`:**
   - if $b = 1$, the error propagates to the caller: a single item does not fit;
   - otherwise call `empty_cache()` exactly once, set `size` to $\lfloor b / 2 \rfloor$
     (half of the batch that failed), and go back to step 1 with the same `start`.
3. **If the model returns**, keep its output and advance `start` by $b$.

**The reduced size is kept** for all later batches when `grow_after` is `None`. When
`grow_after` is a number $g$, a reduced size grows back: after $g$ consecutive
successful batches at a size below `batch_size`, `size` becomes
$\min(2 \cdot \text{size},\ \text{batch\_size})$ and the count starts again at zero. An
out-of-memory error also sets the count to zero. Batches that succeed at
`batch_size` are not counted.

**When `empty_cache()` runs, the failed batch must be unreachable.** The batch passed
to the model stands for everything the failed call allocated. At the moment
`empty_cache()` is called, nothing may refer to it any more: not a variable of your
function, and not the exception, whose traceback keeps the frames of the failed call
(and every tensor in them) alive for as long as the exception is being handled.
`empty_cache()` is called only after a failure that is retried: never after a
successful batch, and not before re-raising.

Any exception that is not a `torch.OutOfMemoryError` propagates unchanged, at once.
`items` is not modified.

## Shapes

| name | shape | dtype |
|---|---|---|
| `items` | `(N, ...)` | any |
| `model(batch)` for a batch of `b` rows | `(b, ...)` | any |
| result | `(N, ...)` with the model's trailing dimensions | the model's |

## Example

Ten items, `batch_size = 8`, and a model that has memory for three rows:

```python
run_batched(model, items, 8, empty_cache)
```

| batch tried | rows | outcome | then |
|---|---|---|---|
| `items[0:8]` | 8 | out of memory | `empty_cache()`, size 4 |
| `items[0:4]` | 4 | out of memory | `empty_cache()`, size 2 |
| `items[0:2]` | 2 | ok | |
| `items[2:4]` | 2 | ok | |
| `items[4:6]`, `[6:8]`, `[8:10]` | 2 | ok | |

The model is called 7 times with batches of 8, 4, 2, 2, 2, 2, 2 rows, `empty_cache`
twice, and the result has 10 rows in the order of `items`.

With `grow_after=2` the same run tries to grow after every two successes:

| batch tried | rows | outcome | then |
|---|---|---|---|
| `items[0:8]`, `items[0:4]` | 8, 4 | out of memory | as above, size 2 |
| `items[0:2]`, `items[2:4]` | 2, 2 | ok, ok | two successes: size 4 |
| `items[4:8]` | 4 | out of memory | `empty_cache()`, size 2, count 0 |
| `items[4:6]`, `items[6:8]` | 2, 2 | ok, ok | two successes: size 4 |
| `items[8:10]` | 2 | ok | only two items were left |

Batches of 8, 4, 2, 2, 4, 2, 2, 2 rows and three calls of `empty_cache`. Against a
model whose limit never changes, growing back only buys failures; it pays off when
the failure was transient, such as another request's memory spike.

## Constraints

- $1 \le N \le 40$, $1 \le \text{batch\_size} \le 64$, $1 \le \text{grow\_after} \le 4$.
- The result is compared with one call of the model's function on all items, to a
  relative tolerance of `1e-5` and an absolute tolerance of `1e-6`.
- The sequence of batch sizes tried is compared exactly, and so is the number of
  `empty_cache` calls (reported as `empty_cache_calls`): one per failure that is
  retried.
- The judge's model refuses to be called more than 80 times.
- **What runs where.** The cases without a hardware mark run on the CPU with a model
  that raises the real `torch.OutOfMemoryError` above a row limit (and, in some
  cases, once on a chosen call although the batch would fit), logs every call, and
  holds a weak reference to each failed batch to see whether it is still alive when
  `empty_cache` runs. One case runs only on a machine with an NVIDIA GPU: the same
  model and checks with `items` on the CUDA device and `torch.cuda.empty_cache`
  called for real. It does not provoke a real out-of-memory error, whose threshold
  depends on the machine. The GPU case was written from the PyTorch 2.15
  documentation and could not be run when the problem was authored.

## Why this matters

A server that batches requests does not know in advance how much memory a batch
needs: it depends on the longest prompt, the number of images, the beam width. An
out-of-memory error is therefore an ordinary event, and the difference between an
engine that survives it and one that dies is this function. The classic failure is a
handler that retries inside the `except` block: the exception still pins the
activations of the attempt that failed, the retry fails for lack of the very memory
that it was supposed to reuse, and the process ends with a second out-of-memory
error raised "during handling of the above exception". The quieter failures are
catching `RuntimeError` (which turns a shape bug into an endless series of smaller
batches), retrying from the beginning (duplicate outputs), and calling
`empty_cache()` after every batch (which makes every batch pay for fresh
allocations).
