Write three sums of products over named axes: attention scores, a bilinear form per
batch row, and a projection with a different matrix for every head. Each is one line
once its index equation is written down, and the equation is where the mistakes are:
which axes are paired, which are summed, and in what order the rest come out.

## Specification

```python
def attention_scores(q, k): ...
def bilinear(x, w, y): ...
def per_head_project(x, w): ...
```

**`attention_scores(q, k)`** takes queries `q` of shape `(B, H, T, D)` and keys `k` of
shape `(B, H, S, D)`, and returns the dot product of every query with every key of the
same batch row and head, shape `(B, H, T, S)`:

$$
\text{out}[b, h, t, s] = \sum_{d} q[b, h, t, d]\; k[b, h, s, d] .
$$

The query position comes before the key position. The scores are not divided by
$\sqrt{D}$.

**`bilinear(x, w, y)`** takes `x` of shape `(B, D)`, a matrix `w` of shape `(D, E)`,
and `y` of shape `(B, E)`, and returns one number per row, shape `(B,)`:

$$
\text{out}[b] = \sum_{d} \sum_{e} x[b, d]\; w[d, e]\; y[b, e] .
$$

Row `b` of `x` meets row `b` of `y` and no other row.

**`per_head_project(x, w)`** takes `x` of shape `(B, H, T, D)` and one matrix per
head, `w` of shape `(H, D, E)`, and returns shape `(B, H, T, E)`:

$$
\text{out}[b, h, t, e] = \sum_{d} x[b, h, t, d]\; w[h, d, e] .
$$

Head `h` is multiplied by `w[h]`.

- Every input is a float32 tensor, and every function returns a float32 tensor.

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

`einsum(equation, *operands)` (`torch.einsum`, `jnp.einsum`, `mx.einsum`) computes
any sum of products from its index equation. The equation has one group of letters
per operand, one letter per axis, then `->` and the letters of the result:

- A letter names an axis. The same letter on two operands means the same index: the
  two axes are walked together and must have the same size.
- A letter that is missing after `->` is summed over.
- The letters after `->` are the axes of the result, in that order.

So `"ij,j->i"` is a matrix times a vector, $\text{out}[i] = \sum_j a[i, j]\, v[j]$, and
`"bi,bj->bij"` is an outer product per row, with nothing summed.

Matrix multiplication is the special case that the `@` operator covers: `a @ b` sums
the last axis of `a` against the second-to-last axis of `b`, and broadcasts all the
axes before those as in Broadcasting on Purpose. A contraction that does not have
that layout needs its axes moved first (`transpose`, `swapaxes`, or `None`).

> [!WARNING]
> Nothing checks that the letters mean what you intended. Letters in the wrong order
> after `->`, or on an operand whose two axes have the same size, give a transposed
> result of the right shape.

Either tool is a correct way to solve this problem.

## Example

```python
q = [[[[1.0, 0.0], [0.0, 1.0]]]]  # shape (1, 1, 2, 2): T = 2
k = [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]]  # shape (1, 1, 3, 2): S = 3
attention_scores(q, k)
# [[[[ 1.0,  2.0,  3.0],            # query 0 reads feature 0 of each key
#    [10.0, 20.0, 30.0]]]]          # shape (1, 1, 2, 3): one row per query

x = [[1.0, 2.0], [0.0, 1.0]]  # shape (2, 2)
w = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # shape (2, 3)
y = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]  # shape (2, 3)
bilinear(x, w, y)
# [24.0, 5.0]                       # shape (2,)

x = [[[[1.0, 0.0]], [[0.0, 1.0]]]]  # shape (1, 2, 1, 2): H = 2
w = [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # the matrix of head 0
    [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],  # the matrix of head 1
]  # shape (2, 2, 3)
per_head_project(x, w)
# [[[[ 1.0,  2.0,  3.0]],           # head 0: row 0 of w[0]
#   [[40.0, 50.0, 60.0]]]]          # head 1: row 1 of w[1]; shape (1, 2, 1, 3)
```

For `bilinear`, row 0 is `[1, 2] @ w = [9, 12, 15]`, then `[9, 12, 15]` times
`[1, 0, 1]` summed, 24. Row 1 is `[4, 5, 6]` times `[0, 1, 0]` summed, 5.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`, which accepts the sums taken in any order.
- Every size is between 1 and 8. Any of them can be 1, and any two can be equal.
- Entries lie in $[-2, 2]$ in the random cases; the cases with hand-written values
  use whole numbers up to 1000.
- No Python loops are needed.

## Why this matters

`attention_scores` is the score computation of Causal Self-Attention and of
Grouped-Query Attention, where `T` and `S` differ as soon as there is a KV cache. Caching Latents
with Multi-Head Latent Attention applies one matrix per head with
`"hcq,btq->bhtc"`, the same pairing as `per_head_project`. One Step of the Gated
Delta Rule reads a per-head memory matrix with `"hvk,hk->hv"`, and Multiplying
Without Dequantizing sums group by group with `"...gk,ogk->...og"`. A model's test
shapes are often square (as many queries as keys, a square projection), and that is
where a transposed contraction returns the right shape and wrong numbers.
