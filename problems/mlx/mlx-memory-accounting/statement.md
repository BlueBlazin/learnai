Write the two tools every memory investigation on Apple silicon starts with: a
prediction of how many bytes a set of arrays needs, and a measurement of how much MLX
memory one function call really used. The arithmetic is trivial. The measurement is
not, because MLX is lazy (a result that was never evaluated has allocated nothing)
and because its counters are global to the process.

## Specification

```python
def predicted_bytes(shapes, dtype): ...
def measure(fn, *args): ...
```

**`predicted_bytes(shapes, dtype)`** returns the number of bytes needed to store
arrays of the given shapes, all of dtype `dtype`:

$$
\text{bytes} = \text{itemsize}(\text{dtype}) \cdot \sum_{s \,\in\, \text{shapes}} \; \prod_{d \,\in\, s} d .
$$

- `shapes`: a list of shapes, each a tuple of non-negative ints. The empty tuple `()`
  is a scalar (one element); a shape containing a 0 is an empty array (no elements);
  the list may be empty.
- `dtype`: an MLX dtype (`mx.float32`, `mx.float16`, `mx.bfloat16`, `mx.int8`,
  `mx.bool_`, ...).
- Returns a Python `int`.

**`measure(fn, *args)`** calls `fn(*args)`, makes sure the work is really done, and
returns a dict with exactly these keys:

| key | value |
|---|---|
| `"result"` | what `fn` returned (an array, or a tuple, list, or dict of arrays), **evaluated** |
| `"peak_bytes"` | the most MLX memory in use at any one time during the call, minus what was in use just before it |
| `"active_bytes"` | the MLX memory in use after the call, minus what was in use just before it: what the call left allocated, which is its result |
| `"cache_bytes"` | the size of MLX's buffer cache after the call: buffers the call allocated and freed, which MLX keeps for reuse |

The three counts are Python `int`s. They must be attributable to `fn`:

- **The result is evaluated inside the measurement.** `fn` returns a lazy graph.
  Until it is evaluated, nothing has been allocated and every counter reads as if
  nothing happened.
- **The arguments are not charged to `fn`.** Memory the arguments occupy is in use
  before the call. An argument may itself be lazy; the memory needed to compute and
  hold it is the caller's, not `fn`'s. Arguments that are not arrays (Python numbers)
  are passed through.
- **Earlier work is not charged to `fn`.** The process has a history: a high-water
  mark from before, and freed buffers sitting in the cache. Neither may show up in the
  report.

The MLX functions involved (MLX 0.32.4; all in `mlx.core`):

| function | what it does |
|---|---|
| `mx.get_active_memory()` | bytes in use by arrays that are alive; excludes the cache |
| `mx.get_peak_memory()` | the largest value active memory has had since the last reset |
| `mx.reset_peak_memory()` | resets that high-water mark |
| `mx.get_cache_memory()` | bytes of freed buffers MLX is holding for reuse |
| `mx.clear_cache()` | returns the cached buffers to the system |
| `mx.synchronize()` | waits until the GPU has finished everything submitted so far |

> [!NOTE]
> The buffers of a graph's temporaries are released when the GPU reports completion,
> which can be a moment *after* `mx.eval` returns. Until then they still count as
> active. Call `mx.synchronize()` after evaluating and before reading the counters.

## Shapes

| name | shape | dtype |
|---|---|---|
| `shapes` | list of tuples of ints | Python `int` |
| result of `predicted_bytes` | scalar | Python `int` |
| `"peak_bytes"`, `"active_bytes"`, `"cache_bytes"` | scalar | Python `int` |

## Example

```python
predicted_bytes([(2048, 2048)], mx.float32)  # 16_777_216 (16 MiB)
predicted_bytes([(2048, 2048)], mx.float16)  # 8_388_608
predicted_bytes([(4096, 1024), (1024,)], mx.float16)  # 8_390_656
# The keys and values of one layer: 8 heads, 4096 tokens, head dimension 64.
predicted_bytes([(1, 8, 4096, 64)] * 2, mx.bfloat16)  # 8_388_608
```

```python
x = mx.random.normal((2048, 2048))  # 16 MiB of float32
mx.eval(x)

measure(lambda a: a * 2.0, x)
# peak_bytes   16_777_220     the result, and 4 bytes for the constant 2.0
# active_bytes 16_777_216     the result is still alive: the dict holds it
# cache_bytes           4


def keeps_half(a):
    temporary = a * 2.0  # 16 MiB
    mx.eval(temporary)
    return temporary[:1024] + 1.0  # 8 MiB


measure(keeps_half, x)
# peak_bytes   25_165_832     24 MiB: the temporary and the result exist together
# active_bytes  8_388_608      8 MiB: only the result is left
# cache_bytes  16_777_224     16 MiB: the temporary's buffer, freed and kept for reuse
```

`x` is 16 MiB and in use throughout, and appears in none of the numbers. Without the
evaluation inside `measure`, the first call would report 0 for all three.

## Constraints

- `predicted_bytes` is compared exactly.
- `"result"` is compared with a NumPy computation (relative tolerance `1e-4`,
  absolute `1e-6`) and must have no pending operations.
- Each byte count must be within **10 percent of the expected value plus 1 MiB**,
  where the expected value is the size of the arrays the fixture allocates, as in the
  example. The fixtures allocate between 8 and 32 MiB, so array sizes dominate any
  allocator overhead. These cases need the Apple GPU.
- Before each call of `measure`, the judge does unrelated MLX work: it allocates,
  evaluates, and frees a 64 MiB array. Some cases call `measure` twice in a row.
- `fn` may call `mx.eval` itself. `fn` and its arguments are not modified.

## Why this matters

On Apple silicon the GPU has no memory of its own: weights, the KV cache, and
activations share the machine's RAM with everything else, and the first question
about any model or optimization is whether it fits. The prediction side is how you
size a KV cache before allocating it (layers × 2 × heads × tokens × head dimension ×
itemsize). The measurement side is how you find out that the prediction was wrong,
and both classic ways of fooling yourself are in this problem: timing or measuring a
lazy graph that never ran, which makes every optimization look free, and reading
process-wide counters without a baseline, which attributes the model's own 4 GB to
whatever you measured after loading it. `active`, `cache`, and `peak` are also three
different answers to "how much memory does it use": what your arrays hold, what MLX
is keeping from the system on top of that, and the worst moment, which is the number
that decides whether the machine starts swapping.
