Implement a batched linear layer where each row in the batch can use a different LoRA adapter, or no adapter at all. Apply the shared base weight once to the entire batch, then add each row's low-rank adapter update without merging adapters into the base weight.

## Specification

```python
def batched_lora_forward(x, base, a_stack, b_stack, ranks, adapter_ids, alpha): ...
```

A server holds one frozen base linear layer with weight $W$ of shape `(out, in)` and $N$ LoRA adapters, padded to a maximum rank $r_{\max}$. Adapter $n$ has true rank $r_n$ ($1 \le r_n \le r_{\max}$), a down-projection matrix $A_n$ of shape `(r_n, in)`, and an up-projection matrix $B_n$ of shape `(out, r_n)`. For an input vector $x$ of length `in` and a shared scaling factor $\alpha$, the base layer computes $x W^{\top}$, and adapter $n$ adds a scaled low-rank update:

$$
x W^{\top} + \frac{\alpha}{r_n}\,\big(x A_n^{\top}\big) B_n^{\top}.
$$

The function receives the following arguments (see the Shapes table below for exact dimensions and dtypes):

- **`x`**: Input activations for a batch of $B$ requests, either 2D (`(B, in)`) or 3D (`(B, T, in)`) for $B$ sequences of $T$ tokens. Axis 0 is the batch axis indexed by `adapter_ids`, so every token in sequence $b$ uses `adapter_ids[b]`.
- **`base`**: Callable representing the frozen base layer. Calling `base(x)` returns $x W^{\top}$ for any `x` with trailing dimension `in`. You do not have direct access to $W$, and each call to `base` is expensive.
- **`a_stack`** and **`b_stack`**: Stacked adapter weights, zero-padded to $r_{\max}$. Rows $r_n, \dots, r_{\max} - 1$ of `a_stack[n]` and columns $r_n, \dots, r_{\max} - 1$ of `b_stack[n]` are zero.
- **`ranks`**: True rank $r_n$ of each adapter, with $1 \le r_n \le r_{\max}$.
- **`adapter_ids`**: Adapter index for each batch row $b$, either an integer in $[0, N)$ or $-1$ to indicate that row $b$ uses the base model alone.
- **`alpha`**: Python float shared across all adapters. Adapter $n$ is scaled by $\alpha / r_n$ using its own rank $r_n$, not $r_{\max}$.

Return a tensor $y$ with the same shape as `base(x)`. For each batch row $b$, with $n = \text{adapter\_ids}[b]$,

$$
y_b =
\begin{cases}
x_b W^{\top} + \dfrac{\alpha}{r_n}\,\big(x_b A_n^{\top}\big) B_n^{\top} & \text{if } n \ge 0, \\[2ex]
x_b W^{\top} & \text{if } n = -1.
\end{cases}
$$

**Cost.** Call `base` exactly once per call to `batched_lora_forward`, passing the full batch `x`.

**Speed.** Compute the adapter updates with batched tensor operations rather than a Python loop over the rows of the batch.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, in)` or `(B, T, in)` | float32 |
| `base(x)` | `(B, out)` or `(B, T, out)` | float32 |
| `a_stack` | `(N, r_max, in)` | float32 |
| `b_stack` | `(N, out, r_max)` | float32 |
| `ranks` | `(N,)` | integer |
| `adapter_ids` | `(B,)` | integer |
| result | `(B, out)` or `(B, T, out)` | float32 |

## Example

This example reuses the base layer and adapter 0 from `lora-merge`, and adds a rank-1 adapter 1 padded to rank 2.

```python
w = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]  # out = 2, in = 3; base(x) = x @ w.T
a_stack = [
    [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],  # adapter 0, rank 2
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # adapter 1, rank 1 (row 1 is padding)
]
b_stack = [
    [[1.0, 0.0], [2.0, -1.0]],           # adapter 0, rank 2
    [[1.0, 0.0], [1.0, 0.0]],            # adapter 1, rank 1 (col 1 is padding)
]
ranks = [2, 1]
alpha = 2.0
x = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]
adapter_ids = [0, -1, 1, 0]

batched_lora_forward(x, base, a_stack, b_stack, ranks, adapter_ids, alpha)
# [[11.0,  3.0],
#  [ 7.0, -1.0],
#  [ 9.0,  1.0],
#  [ 0.0, -1.0]]
```

The first three rows share the same input `[1, 2, 3]` (whose base output is `[7, -1]`) and produce three different outputs:

- **Row 0 (adapter 0, scale $2/2 = 1$).** $x A_0^{\top} = [4, 4]$ and $[4, 4] B_0^{\top} = [4, 4]$, so $y_0 = [7, -1] + [4, 4] = [11, 3]$.
- **Row 1 (no adapter).** $y_1 = [7, -1]$. Indexing `a_stack` directly with `-1` would mistakenly apply adapter 1.
- **Row 2 (adapter 1, scale $2/1 = 2$).** $x A_1^{\top} = [1, 0]$ and $[1, 0] B_1^{\top} = [1, 1]$, so $y_2 = [7, -1] + 2 \cdot [1, 1] = [9, 1]$. Scaling by $r_{\max} = 2$ instead of $r_1 = 1$ would give `[8, 0]`.
- **Row 3 (adapter 0, scale $2/2 = 1$).** Base output is `[0, 1]`, $x A_0^{\top} = [0, 2]$, and $[0, 2] B_0^{\top} = [0, -2]$, so $y_3 = [0, 1] + [0, -2] = [0, -1]$.

## Constraints

- $1 \le B \le 512$, $1 \le T \le 16$, $1 \le \text{in}, \text{out} \le 128$, $1 \le N \le 16$, and $1 \le r_n \le r_{\max} \le 8$.
- Tensor entries are of order 1, and $|\alpha| \le 64$.
- `adapter_ids` and `ranks` use the framework's default integer dtype (`int64` in PyTorch, `int32` in JAX and MLX).
- Outputs are checked against the reference and a row-by-row evaluation of the formula with relative tolerance `1e-4` and absolute tolerance `1e-4`.
- **Cost:** `base` may be called at most **1** time and given at most $B$ rows. Looping over rows calls it $B$ times, and splitting the batch by adapter calls it once per active adapter.
- **Speed:** On 512 rows of width 64 across 16 adapters (ranks 1 to 8), your implementation must run within **5x** of the reference in the same framework. Looping over rows in Python is roughly 16x slower in PyTorch and over 100x slower in JAX and MLX. Looping over the *adapters* with one masked update per adapter is fast enough.
- Do not decorate the function with `jax.jit` or `mx.compile`, because `base` tracks its call count.

## Why this matters

A LoRA adapter takes a few megabytes, while the base model takes tens of gigabytes. A multi-tenant server hosting hundreds of fine-tuned adapters cannot store a separate merged model per customer, nor can it run each adapter in its own tiny batch without underutilizing the GPU.

The unmerged form solves this by separating the shared computation $x W^{\top}$ from the lightweight, per-row low-rank updates. Merging adapters into $W$ is fast for a single adapter, but fails for mixed batches because switching weights per request would rewrite the full matrix on the serving path. Mistakes here fail silently. Indexing with `-1` secretly applies the last adapter to a base-model request, and scaling by $r_{\max}$ weakens lower-rank adapters, producing fluent text from the wrong model.
