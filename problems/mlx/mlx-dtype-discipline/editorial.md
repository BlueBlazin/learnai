## The idea

MLX follows NumPy-style type promotion between *arrays* and treats Python scalars as
weakly typed:

- float16 array with float32 array gives float32. There is no warning.
- float16 array with a Python `float` or `int` gives float16. The scalar is converted
  to the array's dtype.

So the dtype of a block is decided by how its constants enter the graph. Written as
Python numbers, they adapt to whatever the weights are; the same code then serves a
float16, a bfloat16, and a float32 checkpoint. Written as `mx.array(...)`, or produced
by a constructor with a default dtype (`mx.zeros`, `mx.ones`, `mx.arange` with a float
argument, `mx.linspace`, `mx.tri`, `mx.random.*`), they are float32 arrays and promote
everything downstream of them.

One documented surprise in the other direction: NumPy scalars and NumPy arrays used
directly as operands do *not* promote in MLX 0.32.4 (`x * np.float32(0.5)` and
`x * np.ones(4, dtype=np.float32)` are float16 for a float16 `x`). They are converted
to the dtype of the MLX operand. Wrapping them first, `mx.array(np.ones(4))`, gives a
float32 array (a float64 NumPy array becomes float32, since the GPU has no float64),
and that promotes.

The epsilon is the mirror image of the problem. Converted to float16, `1e-5` becomes
`1.0014e-05` (a subnormal: float16's smallest normal number is `6.1e-05`, its
smallest subnormal `6e-08`). `1e-6` and `1e-7` survive as `1.013e-06` and `1.19e-07`,
with one or two significant bits. `1e-8` is below half the smallest subnormal and
becomes exactly `0`. Model configurations written for float16 inference use `1e-5` or
`1e-6` for this reason.

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

Nothing in it names a dtype. `mx.fast.rms_norm(x, params["gain"], 1e-5)` and
`mlx.nn.gelu_approx` compute the same two steps with fewer temporaries (a peak of 20
MiB instead of 30 MiB on the first memory case) and are equally accepted.

## What the promotion costs

Peak MLX memory above the inputs for $N = 4096$, $D = 256$, $H = 1024$, measured on
this machine (the hidden activations are 8 MiB in float16):

| version | result dtype | peak |
|---|---|---|
| reference | float16 | 30 MiB |
| `eps = mx.array(1e-5)` | float32 | 64 MiB |
| the same, then `.astype(mx.float16)` | float16 | 66 MiB |
| `mx.sqrt(mx.array(2 / math.pi))` in the GELU, then `.astype` | float16 | 71 MiB |
| parameters and input cast to float32, result cast back | float16 | 68 MiB |

The numbers repeat exactly from run to run: MLX's allocator is deterministic for a
fixed graph, which is what makes a memory limit usable as a test. Note what the limit
can and cannot see. It bounds the large activations; a promotion confined to the last
addition costs 10 MiB here and passes. The dtype check on the result catches that one
unless it is hidden by a cast, so in your own code the reliable tool is the one from
the hints: assert the dtype where the large arrays are made.

## Common mistakes

- **A constant made with `mx.array`.** `mx.array(1e-5)` is a float32 array. Added to
  the mean of squares it makes the RMS float32, then the normalized input, then every
  hidden activation and the output.
- **A constant computed with MLX.** `mx.sqrt(mx.array(2 / math.pi))` is the same
  mistake one step later; use `math.sqrt`. A mask built as
  `mx.where(cond, 0.0, -math.inf)` is float32 for the same reason: neither branch has
  a dtype to adopt.
- **Hiding the promotion with a final `astype`.** The result is float16 and every
  check on the output passes. The activations were float32 while they existed: twice
  the peak memory, and twice the bytes through every operation.
- **Computing in float32 "for accuracy" and casting back.** A deliberate version of
  the same thing. If a step really needs float32 (MLX's fused attention kernel runs
  its softmax in float32 internally for this reason), it should be that step, on a
  small array, not the whole block.
- **An epsilon that underflows.** `1e-8` is `0` in float16. For ordinary rows nothing
  changes, which is why it survives testing; a zero row gives `0 / sqrt(0)`, and one
  NaN in a residual stream spreads to every later position that attends to it.
- **Normalizing by the RMS of the whole batch.** `mx.mean(x * x)` without an axis
  couples rows that should be independent; it passes any test whose rows have similar
  scales.

## Complexity

Two matrix products, $O(N D H)$ time. Peak memory is a small multiple of the hidden
activations, $N \cdot H \cdot 2$ bytes each in float16.

## In the wild

`mlx-lm` model code does not name a float dtype in the ordinary forward pass, so the
same code runs a float16, bfloat16, or float32 checkpoint; where a model computes
something on the side (expert scores, a gate), it casts it to the dtype of the
activations before combining them, as in `scores.astype(x.dtype)`. The check worth
automating in your own loops is the dtype of the KV cache after the first step: it is
the longest-lived array of a generation, and a promotion there is paid for on every
token until the conversation ends.
