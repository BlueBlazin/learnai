Implement three tensor contractions: attention scores between queries and keys, a bilinear form evaluated per batch row, and a linear projection that applies a separate weight matrix to each attention head. Each function computes a sum of products over named axes, where the pairing and output order of those axes determine the result.

## Specification

```python
def attention_scores(q, k): ...
def bilinear(x, w, y): ...
def per_head_project(x, w): ...
```

**`attention_scores(q, k)`** computes dot products between queries and keys within each batch row and attention head. Here `B` is the batch size, `H` is the number of heads, `T` is the number of query positions, `S` is the number of key positions, and `D` is the head dimension. For every batch index `b`, head `h`, query position `t`, and key position `s`, the output is the dot product over `D`:

$$
\text{out}[b, h, t, s] = \sum_{d} q[b, h, t, d]\; k[b, h, s, d] .
$$

The query position axis comes before the key position axis. Do not divide the scores by $\sqrt{D}$.

**`bilinear(x, w, y)`** evaluates a bilinear form for each batch row independently. Here `x` holds `B` vectors of size `D`, `w` is a `(D, E)` weight matrix, and `y` holds `B` vectors of size `E`. For each batch index `b`, the function returns one scalar:

$$
\text{out}[b] = \sum_{d} \sum_{e} x[b, d]\; w[d, e]\; y[b, e] .
$$

Row `b` of `x` pairs only with row `b` of `y`, never with other batch rows.

**`per_head_project(x, w)`** projects a tensor of `B` batch rows, `H` heads, `T` positions, and feature dimension `D` into output dimension `E` using one `(D, E)` matrix per head:

$$
\text{out}[b, h, t, e] = \sum_{d} x[b, h, t, d]\; w[h, d, e] .
$$

Each head `h` uses its own matrix `w[h]`, shared across all batch rows and sequence positions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, H, T, D)` | float32 |
| `k` | `(B, H, S, D)` | float32 |
| result of `attention_scores` | `(B, H, T, S)` | float32 |
| `x` in `bilinear` | `(B, D)` | float32 |
| `w` in `bilinear` | `(D, E)` | float32 |
| `y` | `(B, E)` | float32 |
| result of `bilinear` | `(B,)` | float32 |
| `x` in `per_head_project` | `(B, H, T, D)` | float32 |
| `w` in `per_head_project` | `(H, D, E)` | float32 |
| result of `per_head_project` | `(B, H, T, E)` | float32 |

## Tools

`einsum(equation, *operands)` (`torch.einsum`, `jnp.einsum`, `mx.einsum`) computes any sum of products from an index equation. The equation lists one group of letters per operand (one letter per axis), followed by `->` and the letters of the output:

- Each letter names an axis. Using the same letter on two operands links those axes, so they are indexed together and must have the same size.
- Any letter omitted after `->` is summed over.
- The letters after `->` define the output axes in that exact order.

For example, `"ij,j->i"` multiplies a matrix by a vector, $\text{out}[i] = \sum_j a[i, j]\, v[j]$, while `"bi,bj->bij"` computes an outer product per row without summing any axis.

The `@` operator handles the special case of matrix multiplication. Writing `a @ b` sums the last axis of `a` against the second-to-last axis of `b` and broadcasts all leading axes. Contractions that do not match this layout require reordering or adding axes first with `transpose`, `swapaxes`, or `None`.

> [!WARNING]
> Nothing checks that your index letters match your intent. Putting letters in the wrong order after `->`, or swapping two axes that happen to have the same size, silently produces a transposed result with the expected shape.

Either `einsum` or `@` is a valid way to solve this problem.

## Example

```python
q = [[[[1.0, 0.0], [0.0, 1.0]]]]  # shape (1, 1, 2, 2), T = 2
k = [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]]  # shape (1, 1, 3, 2), S = 3
attention_scores(q, k)
# [[[[ 1.0,  2.0,  3.0],            # query 0 selects feature 0 of each key
#    [10.0, 20.0, 30.0]]]]          # shape (1, 1, 2, 3), one row per query

x = [[1.0, 2.0], [0.0, 1.0]]  # shape (2, 2)
w = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # shape (2, 3)
y = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]  # shape (2, 3)
bilinear(x, w, y)
# [24.0, 5.0]                       # shape (2,)

x = [[[[1.0, 0.0]], [[0.0, 1.0]]]]  # shape (1, 2, 1, 2), H = 2
w = [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # matrix for head 0
    [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],  # matrix for head 1
]  # shape (2, 2, 3)
per_head_project(x, w)
# [[[[ 1.0,  2.0,  3.0]],           # head 0 selects row 0 of w[0]
#   [[40.0, 50.0, 60.0]]]]          # head 1 selects row 1 of w[1], shape (1, 2, 1, 3)
```

In `bilinear`, row 0 computes `[1, 2] @ w = [9, 12, 15]`, then takes its dot product with `[1, 0, 1]` to get `24.0`. Row 1 computes `[0, 1] @ w = [4, 5, 6]`, then takes its dot product with `[0, 1, 0]` to get `5.0`.

## Constraints

- Results are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-4`, which allows floating-point sums in any order.
- Every dimension size is between `1` and `8`. Any dimension can be `1`, and any two dimensions can be equal.
- Random test inputs have entries in $[-2, 2]$. Hand-written test cases use whole numbers up to `1000`.
- No Python loops are needed.

## Why this matters

`attention_scores` is the core score computation in causal self-attention and grouped-query attention, where `T` and `S` differ as soon as a KV cache is used. Multi-head latent attention applies one matrix per head with `"hcq,btq->bhtc"`, the same head-pairing pattern as `per_head_project`. Similarly, the gated delta rule reads a per-head memory matrix with `"hvk,hk->hv"`, and grouped quantized matrix multiplication contracts group by group with `"...gk,ogk->...og"`.

Unit tests often use square shapes, such as equal query and key lengths or square projection matrices. On those shapes, a transposed contraction returns the right shape with completely wrong values, causing silent bugs during training or inference.
