Write three functions that write into a tensor at positions given by an index tensor.
A scatter is a gather run backwards: the index says where each value goes. The three
frameworks spell it in three different ways, and a plain write is the wrong tool as
soon as two values go to the same place.

## Specification

```python
def one_hot_rows(ids, V): ...
def token_counts(ids, V): ...
def write_at(x, idx, values): ...
```

**`one_hot_rows(ids, V)`** takes token ids of shape `(B,)` and returns a float32
tensor of shape `(B, V)` with a single 1 in every row:

$$
\text{out}[b, v] = \begin{cases} 1 & \text{if } v = \text{ids}[b], \\ 0 & \text{otherwise.} \end{cases}
$$

Do not call your framework's built-in `one_hot` (`torch.nn.functional.one_hot`,
`jax.nn.one_hot`) anywhere in this problem.

**`token_counts(ids, V)`** takes token ids of shape `(B, T)` and returns an integer
tensor of shape `(B, V)`: how many times each token occurs in each row,

$$
\text{out}[b, v] = \#\{\, t : \text{ids}[b, t] = v \,\} .
$$

A token that occurs three times in a row counts 3. Every row of the result sums to `T`.

**`write_at(x, idx, values)`** takes `x` of shape `(B, V)`, `idx` of shape `(B, K)`,
and `values` of shape `(B, K)`. It returns a new tensor of shape `(B, V)` that equals
`x` except at the named columns of each row:

$$
\text{out}[b,\ \text{idx}[b, k]] = \text{values}[b, k] .
$$

Within a row of `idx` no column is named twice. (If one were, two values would be
written to the same entry, and the frameworks do not define which of them stays.)
`x` itself must be unchanged after the call.

- `ids`, `idx`: integer tensors of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX). Every entry is at least 0 and less than `V`.
- `V`: a positive Python int.
- `x`, `values`: float32 tensors.
- `one_hot_rows` and `write_at` return float32. `token_counts` returns an integer
  tensor (any integer width), not a float tensor.

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

A scatter along axis 1 of a two-axis tensor is the write

$$
\text{out}[i,\ \text{index}[i, j]] = \text{src}[i, j] ,
$$

applied to a copy of the input; entries that no index names keep their old value.
Compare it with the gather of Gathering Along an Axis,
$\text{out}[i, j] = \text{input}[i,\ \text{index}[i, j]]$: the index has moved to the
left-hand side. As there, the index needs as many axes as the tensor written into.

- PyTorch: `x.scatter(dim, index, src)` returns a new tensor and `x.scatter_` writes
  into `x`. `src` is a tensor of the index's shape, or a Python number.
- JAX arrays cannot be assigned to. `x.at[rows, cols].set(values)` returns a new array;
  `rows` and `cols` are paired as in Paired and Outer Indexing, so `rows` has to
  broadcast against `cols`.
- MLX: `mx.put_along_axis(x, index, values, axis)` returns a new array. The indexed
  assignment `x[rows, cols] = values` writes into `x`.

> [!WARNING]
> When an index names one entry several times, a setting write stores one of the
> values and drops the rest. The same is true of `x[rows, cols] += 1` in PyTorch and
> MLX: the entry is incremented once. Each framework has a separate adding form:
> `scatter_add` in PyTorch, and `x.at[rows, cols].add(values)` in JAX and in MLX.

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

In `token_counts`, a write that sets a 1 for every id returns `[0, 1, 0, 1]` for
row 0: the second occurrence of token 1 overwrites the first.

## Constraints

- Results are compared with the reference exactly.
- $1 \le B, T, V \le 16$ and $1 \le K \le V$. Any size can be 1, and any two can be
  equal.
- Entries of `x` and `values` lie in $[-1000, 1000]$.
- Do not call your framework's built-in `one_hot` (`torch.nn.functional.one_hot`,
  `jax.nn.one_hot`); a solution that calls it is rejected. Every other function of
  the framework is allowed.
- No Python loops are needed.

## Why this matters

Frequency and Presence Penalties begins with `token_counts` over the generated part
of each row, and its penalty grows with every repeat, so a count that saturates at 1
weakens it without any error. Vectorized Repetition Penalty marks the tokens a row
has seen with a setting scatter, where a repeat is harmless. Nucleus (Top-p)
Filtering makes its decision in sorted order and returns it to token order with
`write_at`. Sorting and Unsorting, later in this module, is built on that last step.
