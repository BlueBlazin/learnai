Write the three steps around every sorted sampling filter: get the order that sorts
each row, invert a permutation, and carry a decision made in sorted order back to
token order. Two things make it more than a call to `argsort`: equal values must be
ranked by index, and going back is a different operation from going there.

## Specification

```python
def rank_descending(x): ...
def inverse_permutation(perm): ...
def keep_to_original_order(keep_sorted, order): ...
```

**`rank_descending(x)`** takes `x` of shape `(B, V)` and returns, for every row, the
column indices that sort the row from largest to smallest, shape `(B, V)`:
`out[b, k]` is the column of the value at rank `k`, so

$$
x[b,\ \text{out}[b, 0]] \ \ge\ x[b,\ \text{out}[b, 1]] \ \ge\ \dots \ \ge\ x[b,\ \text{out}[b, V - 1]] .
$$

Equal values are ranked by column, **lowest column first**. With that rule the result
is unique: every row of it is a permutation of $0, \dots, V - 1$.

**`inverse_permutation(perm)`** takes `perm` of shape `(B, V)`, where every row is a
permutation of $0, \dots, V - 1$, and returns the permutation that undoes each row,
shape `(B, V)`:

$$
\text{out}[b,\ \text{perm}[b, k]] = k .
$$

If `perm[b, k]` is the column at rank `k`, then `out[b, v]` is the rank of column `v`.

**`keep_to_original_order(keep_sorted, order)`** takes a boolean `keep_sorted` of
shape `(B, V)`, where `keep_sorted[b, k]` is a decision about the value at rank `k`
of row `b`, and `order` of shape `(B, V)` as returned by `rank_descending`. It
returns the same decisions indexed by column, shape `(B, V)`:

$$
\text{out}[b,\ \text{order}[b, k]] = \text{keep\_sorted}[b, k] .
$$

- `x`: a float32 tensor of finite values.
- `perm`, `order`: integer tensors of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX).
- `rank_descending` and `inverse_permutation` return integer tensors (any integer
  type; MLX's `argsort` returns uint32, which is accepted).
  `keep_to_original_order` returns a boolean tensor.

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

## Tools

- `argsort(x)` (`torch.argsort`, `jnp.argsort`, `mx.argsort`) returns the indices
  that sort `x` in **ascending** order along one axis. Reading `x` through them, with
  the gather of Gathering Along an Axis, gives the sorted values.
- A sort is **stable** when equal elements keep the order they had in the input.
  `torch.argsort` is not stable unless you pass `stable=True`, and it takes
  `descending=True`. `jnp.argsort` is stable by default. `mx.argsort` is stable, and
  takes neither argument.
- Reading through a permutation and writing through it are opposite operations: one
  is a gather and the other a scatter, as in Scatter and Indexed Writes.

> [!WARNING]
> A descending order is not an ascending order reversed. Reversing a stable ascending
> sort also reverses each group of equal values, which ranks them highest index
> first.

## Example

```python
x = [[0.5, 2.0, 0.5, 3.0], [1.0, 1.0, 4.0, 1.0]]
order = rank_descending(x)
# [[3, 1, 0, 2],          # 3.0, 2.0, then the two 0.5s: column 0 before column 2
#  [2, 0, 1, 3]]          # 4.0, then the three 1.0s in column order

inverse_permutation(order)
# [[2, 1, 3, 0],          # column 0 is at rank 2, column 1 at rank 1, ...
#  [1, 2, 0, 3]]

keep_sorted = [
    [True, True, False, False],  # keep the top two of row 0
    [True, False, False, False],  # keep the top one of row 1
]
keep_to_original_order(keep_sorted, order)
# [[False,  True, False,  True],               # columns 3 and 1
#  [False, False,  True, False]]               # column 2
```

Indexing `keep_sorted` with `order` a second time gives
`[[False, True, True, False], ...]` for row 0, which keeps column 2 (a 0.5) and
drops column 3 (the 3.0).

## Constraints

- Results are compared with the reference exactly.
- $1 \le B \le 16$ and $1 \le V \le 64$. Either can be 1, and they can be equal.
- Entries of `x` lie in $[-100, 100]$. There is no NaN. Rows may consist mostly or
  entirely of equal values, also at the ends of that range. Only exactly equal values
  are ties: two values that differ by `1e-5` are ranked by value, wherever their
  columns are.
- No Python loops are needed.

## Why this matters

Nucleus (Top-p) Filtering sorts the probabilities, decides in sorted order where the
cumulative mass crosses `p`, and writes the result back to token order; its
specification ranks equal probabilities by token index, lowest first. Top-k Filtering
uses the inverse permutation directly: token `v` survives when its rank is below `k`.
Heavy-Hitter KV Eviction ranks positions by score and keeps the best. Ties are
common in all of them (a filter upstream has set many logits to the same value), and a
sort that breaks them differently on another device or another framework gives
different text for the same seed.
