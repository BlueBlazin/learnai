Write three functions that answer, for every position of every row at once, a
question that a loop would answer with a running variable: how much came before here,
where did something first happen, has it happened yet. A cumulative sum is that
running variable as a tensor.

## Specification

```python
def exclusive_cumsum(x): ...
def first_true(mask): ...
def through_first_eos(ids, eos): ...
```

**`exclusive_cumsum(x)`** takes `x` with one to three axes and returns, along the
last axis, the sum of the entries strictly before each position. The result has the
shape of `x`:

$$
\text{out}[\dots, i] = \sum_{j < i} x[\dots, j] , \qquad \text{out}[\dots, 0] = 0 .
$$

**`first_true(mask)`** takes a boolean `mask` of shape `(B, T)` and returns the index
of the first true entry of every row, shape `(B,)`:

$$
\text{out}[b] = \min\{\, t : \text{mask}[b, t] \,\} ,
$$

and `T` for a row that has no true entry.

**`through_first_eos(ids, eos)`** takes token ids of shape `(B, T)` and the id `eos`
of the end-of-sequence token. It returns the boolean mask, shape `(B, T)`, of the
positions of each row up to and including its first `eos`:

$$
\text{out}[b, t] = \big(\text{no } j < t \text{ has } \text{ids}[b, j] = \text{eos}\big) .
$$

The first `eos` of a row is itself true, and everything after it is false, including
any later `eos`. A row without `eos` is all true.

- `x`: a float32 tensor. `exclusive_cumsum` returns float32.
- `mask`: a boolean tensor. `first_true` returns an integer tensor (your framework's
  default integer type; any integer type is accepted).
- `ids`: an integer tensor of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX). `eos`: a Python int.
  `through_first_eos` returns a boolean tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(n,)`, `(B, n)`, or `(B, H, n)` | float32 |
| result of `exclusive_cumsum` | the shape of `x` | float32 |
| `mask` | `(B, T)` | bool |
| result of `first_true` | `(B,)` | integer |
| `ids` | `(B, T)` | integer |
| result of `through_first_eos` | `(B, T)` | bool |

## Tools

- `cumsum(x, axis)` (`torch.cumsum(x, dim=...)`, `jnp.cumsum`, `mx.cumsum`) returns
  the **inclusive** running sum along one axis, in the shape of `x`:
  `out[i] = x[0] + ... + x[i]`.
- The running sum of a mask is a running count: at each position, how many true
  entries there have been so far. Boolean input gives integer output.
- A comparison of a running count with a number is a mask again, and the sum of a
  mask along an axis is a count per row. Chaining these replaces "stop at the first
  one" with arithmetic on whole rows.

> [!WARNING]
> `argmax` of a mask does not tell a row whose first entry is true from a row with
> no true entry: both return 0. (JAX and MLX take the `argmax` of a boolean tensor
> directly. PyTorch raises `NotImplementedError` for one, and returns the same two
> zeros once the mask is cast to an integer type.)

## Example

```python
x = [[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -10.0, 5.0]]
exclusive_cumsum(x)
# [[0.0,  1.0,  3.0, 6.0],                 # the inclusive sum would be 1, 3, 6, 10
#  [0.0, 10.0, 10.0, 0.0]]

mask = [
    [False, True, False, True, False],
    [False, False, False, False, False],
    [True, True, False, False, False],
    [False, False, False, False, True],
]
first_true(mask)
# [1, 5, 0, 4]                             # row 1 has no true entry: T = 5

ids = [[5, 2, 7, 2, 9], [1, 3, 4, 6, 8], [2, 2, 5, 2, 5]]
through_first_eos(ids, eos=2)
# [[ True,  True, False, False, False],    # the first 2 is kept, the second is not
#  [ True,  True,  True,  True,  True],    # no eos
#  [ True, False, False, False, False]]
```

## Constraints

- `first_true` and `through_first_eos` are compared with the reference exactly.
  `exclusive_cumsum` is compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-4`.
- Every size is between 1 and 16. Any of them can be 1, and any two can be equal.
- Entries of `x` lie in $[-10, 10]$. Token ids and `eos` lie in $[0, 1000]$.
- No Python loops are needed.

## Why this matters

Nucleus (Top-p) Filtering keeps a token while the probability mass ranked before it
is below `p`, which is `exclusive_cumsum` of the sorted probabilities; the inclusive
sum drops the token that crosses the threshold. Inverse-CDF Sampling finds the first
cumulative sum above a threshold by counting the ones that are not, the same counting
as in `first_true`. Verifying a Greedy Draft and Ragged Acceptance in a Batch accept
a draft up to its first mismatch with a running product, which is the same idea with
multiplication. And every batched decode loop has rows that end at different steps: the tokens a row produced after its own
`eos` are still in the tensor, and `through_first_eos` is the mask that removes them.
