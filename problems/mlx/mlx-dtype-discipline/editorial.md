## The idea

MLX follows NumPy-style type promotion between arrays, while treating Python scalars as weakly typed.

- Combining a `float16` array with a `float32` array produces `float32` without warning.
- Combining a `float16` array with a Python `float` or `int` produces `float16` because the scalar adopts the array's dtype.

Whether a block stays in half precision depends on how its constants enter the graph. Written as plain Python numbers, constants adapt to the dtype of the weights, so the same code works for `float16`, `bfloat16`, and `float32` checkpoints. Wrapped in `mx.array(...)` or created by a default-dtype constructor (`mx.zeros`, `mx.ones`, `mx.arange` with a float argument, `mx.linspace`, `mx.tri`, `mx.random.*`), constants become `float32` arrays and promote everything downstream.

One detail works in the opposite direction in MLX 0.32.4. NumPy scalars and NumPy arrays used directly as operands do *not* promote `float16` (`x * np.float32(0.5)` and `x * np.ones(4, dtype=np.float32)` both stay `float16`). MLX converts them to the dtype of the MLX operand. Wrapping them in `mx.array(np.ones(4))` first creates a `float32` array (since the Apple GPU has no `float64`), and that does promote.

The normalization epsilon poses the opposite risk. In `float16`, the smallest normal number is `6.1e-05` and the smallest subnormal is `6e-08`. Converting `1e-5` to `float16` gives `1.0014e-05`, which is subnormal but positive. Both `1e-6` and `1e-7` also survive (`1.013e-06` and `1.19e-07`, with one or two significant bits), whereas `1e-8` falls below half the smallest subnormal and rounds to `0`. For this reason, `float16` model configs typically set epsilon to `1e-5` or `1e-6`.

## Reference solution (MLX)

```python
EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)


def forward_f16(params, x):
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    h = (x / rms * params["gain"]) @ params["w_in"].T
    h = 0.5 * h * (1 + mx.tanh(GELU_C * (h + 0.044715 * h * h * h)))
    return x + h @ params["w_out"].T
```

Nothing in this implementation hardcodes a dtype. Using `mx.fast.rms_norm(x, params["gain"], 1e-5)` and `mlx.nn.gelu_approx` computes the same operations with fewer temporaries (peaking at 20 MiB instead of 30 MiB on the first memory test) and is equally accepted.

## What the promotion costs

Peak MLX memory above the inputs for $N = 4096$, $D = 256$, $H = 1024$, where the `float16` hidden activations are 8 MiB:

| version | result dtype | peak |
|---|---|---|
| reference | `float16` | 30 MiB |
| `eps = mx.array(1e-5)` | `float32` | 64 MiB |
| the same, then `.astype(mx.float16)` | `float16` | 66 MiB |
| `mx.sqrt(mx.array(2 / math.pi))` in the GELU, then `.astype` | `float16` | 71 MiB |
| parameters and input cast to `float32`, result cast back | `float16` | 68 MiB |

MLX's allocator is deterministic for a fixed graph, so these peak numbers repeat exactly across runs and make a memory limit reliable as a test. Keep in mind what a memory limit can and cannot detect. It catches promotion in the large hidden activations, whereas a promotion confined to the final residual addition only adds 10 MiB and stays under the limit. The output dtype check catches that final promotion unless a cast hides it, so the best habit in production code is to assert `.dtype` where large arrays are constructed.

## Common mistakes

- **Wrapping a constant in `mx.array`.** `mx.array(1e-5)` creates a `float32` array. Adding it to the mean of squares promotes the RMS, the normalized input, every hidden activation, and the final output to `float32`.
- **Computing a scalar constant with MLX.** `mx.sqrt(mx.array(2 / math.pi))` causes the same promotion inside GELU. Use `math.sqrt` instead. Similarly, `mx.where(cond, 0.0, -math.inf)` produces `float32` because neither branch carries a `float16` dtype.
- **Hiding promotion with a final `.astype`.** Casting the output to `float16` passes output dtype checks, but the intermediate activations still ran in `float32` at double the peak memory and bandwidth.
- **Computing the whole block in `float32` for accuracy.** If a specific operation genuinely requires `float32` (as in MLX's fused attention kernel, which runs softmax internally in `float32`), confine the upcast to that small operation rather than the entire block.
- **Using an epsilon that underflows.** `1e-8` rounds to `0` in `float16`. Ordinary rows are unaffected, so the bug passes basic tests, but a zero row computes `0 / sqrt(0)`. One `NaN` in the residual stream then spreads to every later position that attends to it.
- **Normalizing across the entire batch.** Calling `mx.mean(x * x)` without `axis=-1` couples independent rows. This bug can slip past tests whose rows all have similar magnitudes.

## Complexity

Two matrix multiplications take $O(N D H)$ time. Peak memory is a small constant multiple of the hidden activation size, which is $2 N H$ bytes in `float16`.

## In the wild

Model code in `mlx-lm` avoids hardcoding float dtypes in the forward pass so that one implementation serves `float16`, `bfloat16`, and `float32` checkpoints. Whenever a model computes an auxiliary tensor such as router scores or a gate, it casts that tensor to match the activations (`scores.astype(x.dtype)`) before combining them. In inference loops, the single most valuable automated check is asserting the dtype of the KV cache after the first step, because the cache persists across the entire generation.
