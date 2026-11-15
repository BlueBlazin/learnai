Run one linear layer for a batch in which every row may use a different LoRA
adapter, or none: the shared base weight is applied once to the whole batch, and
each row adds the low-rank update of its own adapter, without merging anything into
the weight. The formula is that of `lora-merge`, per row; the substance is doing it
for all rows at once, with adapters of different ranks and rows that have no adapter.

## Specification

```python
def batched_lora_forward(x, base, a_stack, b_stack, ranks, adapter_ids, alpha): ...
```

A server holds one frozen base layer with weight $W$ of shape `(out, in)` and $N$
LoRA adapters. Adapter $n$ has rank $r_n$, a down-projection $A_n$ of shape
$(r_n, \text{in})$, and an up-projection $B_n$ of shape $(\text{out}, r_n)$, and
changes the layer's output for an input row $x$ from $x W^{\top}$ to

$$
x W^{\top} + \frac{\alpha}{r_n}\,\big(x A_n^{\top}\big) B_n^{\top}.
$$

- **`base`** is the base layer, given as a callable: `base(x)` returns $x W^{\top}$
  for `x` of shape `(..., in)`. You do not get $W$ itself, and a call is expensive:
  it is the full-size matmul that all requests share.
- **`a_stack`**, shape `(N, r_max, in)`, and **`b_stack`**, shape `(N, out, r_max)`,
  hold all adapters, zero-padded to the largest rank $r_\text{max}$: rows
  $r_n, \dots, r_\text{max} - 1$ of `a_stack[n]` and the same columns of
  `b_stack[n]` are zero.
- **`ranks`**, shape `(N,)`, holds the true rank $r_n$ of each adapter,
  $1 \le r_n \le r_\text{max}$.
- **`adapter_ids`**, shape `(B,)`, gives the adapter of each row of the batch: an
  integer in $[0, N)$, or $-1$ for a row that uses the base model alone.
- **`alpha`** is a Python float, the same for all adapters. Adapter $n$ is scaled
  by $\alpha / r_n$, its own rank, not $r_\text{max}$.
- **`x`** has shape `(B, in)`, or `(B, T, in)` for $B$ sequences of $T$ tokens. The
  first axis is the batch axis that `adapter_ids` indexes; every token of sequence
  $b$ uses adapter `adapter_ids[b]`.

The result $y$ has the shape of `base(x)`, and for every row $b$ of the batch, with
$n$ = `adapter_ids[b]`,

$$
y_b =
\begin{cases}
x_b W^{\top} + \dfrac{\alpha}{r_n}\,\big(x_b A_n^{\top}\big) B_n^{\top} & n \ge 0, \\[2ex]
x_b W^{\top} & n = -1 .
\end{cases}
$$

**Cost.** `base` must be called exactly once per call of `batched_lora_forward`,
with the whole batch `x`. **Speed.** The updates must be computed with batched
tensor operations, not in a Python loop over the rows of the batch.

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

The base layer and adapter 0 are those of the example in `lora-merge`; adapter 1 has
rank 1 and is padded to rank 2.

```python
w = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]  # out = 2, in = 3; base(x) = x @ w.T
a_stack = [
    [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],  # adapter 0, rank 2
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
]  # adapter 1, rank 1 (second row is padding)
b_stack = [
    [[1.0, 0.0], [2.0, -1.0]],
    [[1.0, 0.0], [1.0, 0.0]],
]  # second column of adapter 1 is padding
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

The first three rows are the same input and give three different outputs. The base
output of $[1, 2, 3]$ is $[7, -1]$.

- Row 0, adapter 0, scale $2/2 = 1$: $x A_0^{\top} = [4, 4]$,
  $[4, 4]\,B_0^{\top} = [4, 4]$, so $y = [11, 3]$.
- Row 1, no adapter: $y = [7, -1]$. Indexing the stacks with $-1$ would have applied
  adapter 1.
- Row 2, adapter 1, scale $2/1 = 2$: $x A_1^{\top} = [1, 0]$,
  $[1, 0]\,B_1^{\top} = [1, 1]$, so $y = [7, -1] + 2 \cdot [1, 1] = [9, 1]$. With the
  padded rank the scale would be 1 and the result $[8, 0]$.
- Row 3, adapter 0: base $[0, 1]$, $x A_0^{\top} = [0, 2]$,
  $[0, 2]\,B_0^{\top} = [0, -2]$, so $y = [0, -1]$.

## Constraints

- $1 \le B \le 512$, $1 \le T \le 16$, $1 \le \text{in}, \text{out} \le 128$,
  $1 \le N \le 16$, $1 \le r_n \le r_\text{max} \le 8$. Entries are of order 1 and
  $|\alpha| \le 64$.
- `adapter_ids` and `ranks` arrive in the framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX).
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`, and with a row-by-row evaluation of the formula above.
- **Cost:** `base` is called at most **1** time and is given at most $B$ rows. A
  loop over rows calls it $B$ times; splitting the batch by adapter calls it once per
  adapter present.
- **Speed:** for 512 rows of width 64 over 16 adapters with ranks 1 to 8, the
  function must be within **5x** of the reference in the same framework. Adding the
  updates in a Python loop over the rows is about 16x slower in PyTorch and more
  than 100x slower in JAX and MLX. A loop over the *adapters* (one masked update
  per adapter) is not a loop over rows and is fast enough.
- Do not wrap the function in `jax.jit` or `mx.compile`: `base` counts its calls.

## Why this matters

A LoRA adapter is a few megabytes; the model it adapts is many gigabytes. A provider
with a thousand fine-tuned customers cannot keep a thousand merged copies of the
model in GPU memory, and cannot afford a batch per customer either: most adapters
see a request now and then, and a batch of one wastes the GPU. Serving them together
rests on the identity above. The expensive part, $x W^{\top}$, does not depend on the
adapter, so it is computed once for the whole mixed batch; the adapter-specific part
is two thin matmuls per row, which can be gathered by adapter index and batched as
well. Merging, the cheapest option for a single adapter, is exactly what must not
happen here: a merged weight serves one adapter, and switching it per request means
rewriting the full weight matrix on the request path. The bugs are quiet ones. A row
without an adapter that silently gets the last adapter in the stack, or an adapter
scaled by the padded rank, produces fluent output from the wrong model.
