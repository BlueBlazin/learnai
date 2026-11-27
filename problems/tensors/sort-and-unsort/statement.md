Implement three functions used by sorted sampling filters. Given a 2D tensor of score rows, you will compute the descending sort order for each row, invert a permutation, and map a boolean decision from sorted order back to original column order.

## Specification

```python
def rank_descending(x): ...
def inverse_permutation(perm): ...
def keep_to_original_order(keep_sorted, order): ...
```

Each input is a 2D tensor with `B` rows (batch size) and `V` columns (vocabulary size).

**`rank_descending(x)`** returns the column indices that sort each row of `x` from largest to smallest. Entry `out[b, k]` is the column index of the value at rank `k` in row `b`, where `k = 0` is the largest value:

$$
x[b,\ \text{out}[b, 0]] \ \ge\ x[b,\ \text{out}[b, 1]] \ \ge\ \dots \ \ge\ x[b,\ \text{out}[b, V - 1]] .
$$

Equal values are ranked by column index, **lowest column first**. With this tie-breaking rule, the output is unique and every row is a permutation of $0, \dots, V - 1$.

**`inverse_permutation(perm)`** takes `perm`, where each row is a permutation of $0, \dots, V - 1$, and returns the permutation that undoes each row:

$$
\text{out}[b,\ \text{perm}[b, k]] = k .
$$

If `perm[b, k]` is the column at rank `k`, then `out[b, v]` is the rank of column `v`.

**`keep_to_original_order(keep_sorted, order)`** takes `keep_sorted` and `order` (as returned by `rank_descending`). Entry `keep_sorted[b, k]` is a boolean decision for the value at rank `k` of row `b`. The function returns those same decisions indexed by original column:

$$
\text{out}[b,\ \text{order}[b, k]] = \text{keep\_sorted}[b, k] .
$$

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, V)` | float32 |
| result of `rank_descending` | `(B, V)` | integer |
| `perm` | `(B, V)` | integer |
| result of `inverse_permutation` | `(B, V)` | integer |
| `keep_sorted` | `(B, V)` | bool |
| `order` | `(B, V)` | integer |
| result of `keep_to_original_order` | `(B, V)` | bool |

Input `x` contains finite float32 values. Inputs `perm` and `order` use your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX). `rank_descending` and `inverse_permutation` may return any integer type (including `uint32` from MLX's `argsort`).

## Tools

- `argsort(x)` (`torch.argsort`, `jnp.argsort`, `mx.argsort`) returns the indices that sort `x` in **ascending** order along one axis. Gathering `x` along that axis with those indices gives the sorted values.
- A sort is **stable** when equal elements keep their original relative order. `torch.argsort` is not stable unless you pass `stable=True`, and it also accepts `descending=True`. `jnp.argsort` is stable by default. `mx.argsort` is stable and takes neither argument.
- Reading through a permutation is a gather, whereas writing through a permutation is a scatter.

> [!WARNING]
> A descending sort is not an ascending sort reversed. Reversing a stable ascending sort also reverses every group of equal values, which ranks ties highest index first instead of lowest index first.

## Example

```python
x = [[0.5, 2.0, 0.5, 3.0], [1.0, 1.0, 4.0, 1.0]]
order = rank_descending(x)
# [[3, 1, 0, 2],   # 3.0, 2.0, then the two 0.5s with column 0 before column 2
#  [2, 0, 1, 3]]   # 4.0, then the three 1.0s in column order

inverse_permutation(order)
# [[2, 1, 3, 0],   # column 0 is at rank 2, column 1 at rank 1, ...
#  [1, 2, 0, 3]]

keep_sorted = [
    [True, True, False, False],   # keep the top two values of row 0
    [True, False, False, False],  # keep the top one value of row 1
]
keep_to_original_order(keep_sorted, order)
# [[False,  True, False,  True],  # columns 3 and 1
#  [False, False,  True, False]]  # column 2
```

By contrast, gathering `keep_sorted` through `order` a second time gives `[[False, True, True, False], ...]` for row 0. That keeps column 2 (value `0.5`) and drops column 3 (value `3.0`).

## Constraints

- Results are compared with the reference exactly.
- $1 \le B \le 16$ and $1 \le V \le 64$. Either dimension can be 1, and they can be equal.
- Entries of `x` lie in $[-100, 100]$ with no NaN.
- Rows may consist mostly or entirely of equal values, including at the endpoints $-100$ and $100$.
- Only exactly equal values are ties. Two values that differ by `1e-5` must be ranked by value regardless of their columns.
- No Python loops are needed.

## Why this matters

Nucleus (top-p) filtering sorts probabilities, finds where the cumulative mass crosses `p` in sorted order, and writes that mask back to token order. Its specification breaks ties by lowest token index first. Top-k filtering uses the inverse permutation directly, keeping token `v` when its rank is below `k`. Heavy-hitter KV eviction also ranks positions by score and keeps the top entries.

Ties happen often in these filters, for instance when an upstream filter clips many logits to the same value. If a sort breaks ties differently on another device or framework, the same random seed can produce different text.
