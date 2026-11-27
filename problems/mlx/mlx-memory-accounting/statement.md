Implement two MLX memory accounting functions. The first predicts how many bytes a list of array shapes requires, and the second measures the MLX memory actually used by a single function call. Because MLX evaluates operations lazily and tracks memory with process-wide counters, accurate measurement requires forcing evaluation and isolating the call from surrounding state.

## Specification

```python
def predicted_bytes(shapes, dtype): ...
def measure(fn, *args): ...
```

**`predicted_bytes(shapes, dtype)`** computes the total number of bytes required to store a list of arrays with the given shapes and common dtype `dtype`. Multiply the byte size of one element by the sum of element counts across all shapes:

$$
\text{bytes} = \text{itemsize}(\text{dtype}) \cdot \sum_{s \,\in\, \text{shapes}} \; \prod_{d \,\in\, s} d .
$$

- `shapes` is a list of tuples of non-negative integers. The empty tuple `()` is a scalar with 1 element, a shape containing `0` is an empty array with 0 elements, and `shapes` itself may be empty (returning `0`).
- `dtype` is an MLX dtype (`mx.float32`, `mx.float16`, `mx.bfloat16`, `mx.int8`, `mx.bool_`, and so on).
- Returns a Python `int`.

**`measure(fn, *args)`** invokes `fn(*args)`, ensures all computation finishes, and returns a dictionary with exactly these four keys:

| key | value |
|---|---|
| `"result"` | the evaluated return value of `fn` (an array, or a tuple, list, or dict of arrays) |
| `"peak_bytes"` | maximum MLX memory in use at any point during the call, minus the memory in use just before the call |
| `"active_bytes"` | MLX memory in use after the call, minus the memory in use just before the call (what the call left allocated) |
| `"cache_bytes"` | size of MLX's buffer cache after the call (buffers allocated and freed by the call, then kept for reuse) |

All three byte counts must be Python `int`s attributable solely to `fn`:

- **Evaluate the result inside the measurement.** `fn` returns a lazy computation graph. Until it is evaluated, MLX allocates nothing and all counters read zero.
- **Exclude the arguments.** Memory occupied by the arguments is already in use before the call. An argument may itself be lazy, and the memory needed to compute and hold it belongs to the caller rather than `fn`. Non-array arguments (such as Python numbers) pass through unchanged.
- **Exclude earlier work.** The process may already have an earlier peak memory high-water mark and freed buffers sitting in the cache. Neither may appear in the reported counts.

The relevant functions in `mlx.core` (MLX 0.32.4) are:

| function | what it does |
|---|---|
| `mx.get_active_memory()` | bytes currently used by live arrays, excluding the cache |
| `mx.get_peak_memory()` | highest value active memory has reached since the last reset |
| `mx.reset_peak_memory()` | resets the peak memory high-water mark |
| `mx.get_cache_memory()` | bytes of freed buffers MLX is holding for reuse |
| `mx.clear_cache()` | returns cached buffers to the system |
| `mx.synchronize()` | blocks until the GPU finishes all submitted work |

> [!NOTE]
> Buffers for a graph's temporary arrays are released when the GPU reports completion, which can happen slightly *after* `mx.eval` returns. Until then, those buffers still count as active. Call `mx.synchronize()` after evaluating and before reading the counters.

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

Although `x` occupies 16 MiB throughout, it does not appear in any of the reported counts. Without evaluating the result inside `measure`, the first call would report 0 for all three metrics.

## Constraints

- `predicted_bytes` must match the exact byte count.
- `"result"` is compared against a NumPy reference (relative tolerance `1e-4`, absolute tolerance `1e-6`) and must have no pending operations.
- Each byte count must fall within **10 percent of the expected value plus 1 MiB**, where the expected value is the size of the arrays allocated by the test fixture. Fixtures allocate between 8 and 32 MiB so array sizes dominate allocator overhead. These cases require an Apple GPU.
- Before each call to `measure`, the judge allocates, evaluates, and frees an unrelated 64 MiB array. Some test cases call `measure` twice in a row.
- `fn` may call `mx.eval` internally. Do not modify `fn` or its arguments.

## Why this matters

On Apple silicon, the GPU shares system RAM with the CPU, so weights, the KV cache, and activations all compete for the same unified memory pool. Before running a model or an optimization, you first estimate whether it fits using `predicted_bytes` (for example, sizing a KV cache as `layers * 2 * heads * tokens * head_dim * itemsize`).

Measurement tells you when reality disagrees with that estimate. Two classic mistakes invalidate MLX measurements: measuring a lazy graph that never executed (making every operation look free) and reading process-wide counters without a baseline (charging the model's own weights to a small helper function). Tracking `active`, `cache`, and `peak` separately also answers three distinct questions: what your live arrays hold, what MLX retains from the OS for reuse, and the worst-case spike that determines whether the machine starts swapping.
