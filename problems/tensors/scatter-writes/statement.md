Write three functions that write values into a tensor at positions given by an index tensor. A scatter is a gather in reverse, where the index specifies the destination of each value. Because the three frameworks spell indexed writes differently, and because a plain assignment drops duplicates when two values target the same position, you need to choose the right write operation for each case.

## Specification

```python
def one_hot_rows(ids, V): ...
def token_counts(ids, V): ...
def write_at(x, idx, values): ...
```

**`one_hot_rows(ids, V)`** takes a 1D tensor `ids` of `B` token indices and a positive integer vocabulary size `V`. It returns a float32 tensor with a single `1.0` in each row `b` at column `ids[b]` and `0.0` at every other column `v`:

$$
\text{out}[b, v] = \begin{cases} 1 & \text{if } v = \text{ids}[b], \\ 0 & \text{otherwise.} \end{cases}
$$

Do not call your framework's built-in `one_hot` (`torch.nn.functional.one_hot`, `jax.nn.one_hot`) anywhere in this problem.

**`token_counts(ids, V)`** takes a 2D tensor `ids` of `B` sequences of length `T` and a positive integer `V`. It returns an integer tensor counting how many times each token `v` appears in each row `b`:

$$
\text{out}[b, v] = \#\{\, t : \text{ids}[b, t] = v \,\} .
$$

A token that appears three times in a row has count `3`. Every row of the result sums to `T`.

**`write_at(x, idx, values)`** takes a 2D tensor `x` with `B` rows and `V` columns, column indices `idx` selecting `K` columns per row, and replacement entries `values` of the same shape. It returns a new tensor that matches `x` everywhere except at the columns named by `idx`. For each row `b` and selection index `k`:

$$
\text{out}[b,\ \text{idx}[b, k]] = \text{values}[b, k] .
$$

No column is repeated within the same row of `idx`. If a column were repeated, two values would target the same entry and the frameworks do not define which one wins. `x` itself must remain unchanged after the call.

All index tensors (`ids`, `idx`) use your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX), with every entry in $[0, V - 1]$. `V` is a positive Python `int`. `token_counts` may return any integer width, not a float tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `ids` of `one_hot_rows` | `(B,)` | integer |
| result of `one_hot_rows` | `(B, V)` | float32 |
| `ids` of `token_counts` | `(B, T)` | integer |
| result of `token_counts` | `(B, V)` | integer |
| `x` | `(B, V)` | float32 |
| `idx`, `values` | `(B, K)` | integer, float32 |
| result of `write_at` | `(B, V)` | float32 |

## Tools

A scatter along axis 1 of a 2D tensor writes each source entry to the column named by the index:

$$
\text{out}[i,\ \text{index}[i, j]] = \text{src}[i, j] .
$$

This write starts from a copy of the target tensor, so entries that no index names keep their original values. Compare it with the gather from Gathering Along an Axis, $\text{out}[i, j] = \text{input}[i,\ \text{index}[i, j]]$. The index has moved to the left-hand side, and it still needs as many axes as the target tensor.

- **PyTorch:** `x.scatter(dim, index, src)` returns a new tensor, while `x.scatter_` mutates `x` in place. `src` can be a tensor matching `index.shape` or a Python scalar.
- **JAX:** Arrays are immutable. `x.at[rows, cols].set(values)` returns a new array, pairing `rows` and `cols` as in Paired and Outer Indexing so `rows` must broadcast against `cols`.
- **MLX:** `mx.put_along_axis(x, index, values, axis)` returns a new array. Indexed assignment `x[rows, cols] = values` mutates `x` in place.

> [!WARNING]
> When an index names the same entry multiple times, a setting write keeps only one value and drops the rest. In PyTorch and MLX, `x[rows, cols] += 1` has the same behavior and increments a repeated entry only once. Use each framework's accumulating form instead: `scatter_add` in PyTorch, and `x.at[rows, cols].add(values)` in JAX and MLX.

## Example

```python
one_hot_rows([2, 0, 2], V=4)
# [[0.0, 0.0, 1.0, 0.0],
#  [1.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 1.0, 0.0]]                       # shape (3, 4), float32

token_counts([[1, 1, 3], [0, 2, 2]], V=4)
# [[0, 2, 0, 1],                               # row 0: token 1 twice, token 3 once
#  [1, 0, 2, 0]]                               # shape (2, 4), integer

x = [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]  # x[b, v] = 10 b + v
idx = [[3, 0], [1, 2]]
values = [[900.0, 901.0], [910.0, 911.0]]  # values[b, k] = 900 + 10 b + k
write_at(x, idx, values)
# [[901.0,   1.0,   2.0, 900.0],               # 900 to column 3, 901 to column 0
#  [ 10.0, 910.0, 911.0,  13.0]]               # 910 to column 1, 911 to column 2
```

In `token_counts`, a setting write that places `1` at every id returns `[0, 1, 0, 1]` for row 0 because the second occurrence of token 1 overwrites the first.

## Constraints

- Results are compared against the reference for exact equality.
- $1 \le B, T, V \le 16$ and $1 \le K \le V$. Any dimension can be 1, and any two can be equal.
- Entries of `x` and `values` lie in $[-1000, 1000]$.
- Do not call your framework's built-in `one_hot` (`torch.nn.functional.one_hot`, `jax.nn.one_hot`). Solutions that call it are rejected. Every other framework function is allowed.
- No Python loops are needed.

## Why this matters

Decoding penalties and sampling filters rely directly on these writes. Frequency and Presence Penalties starts with `token_counts` over the generated tokens in each row, and its penalty scales with repeat count, so a count that caps at 1 silently weakens the penalty. Vectorized Repetition Penalty marks seen tokens with a setting scatter, where duplicate indices are harmless. Nucleus (Top-p) Filtering computes a mask in sorted order and maps it back to vocabulary order with `write_at`, which Sorting and Unsorting later in this module builds on.
