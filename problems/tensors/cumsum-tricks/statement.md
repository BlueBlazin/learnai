Implement three functions that use cumulative sums to replace left-to-right loops over tensor rows. You will compute the sum strictly before each position, find the index of the first true entry in each row, and build a mask that keeps every token up to and including the first end-of-sequence token.

## Specification

```python
def exclusive_cumsum(x): ...
def first_true(mask): ...
def through_first_eos(ids, eos): ...
```

**`exclusive_cumsum(x)`** takes a 1D, 2D, or 3D tensor `x` whose last axis has length `n` (with optional leading batch/head dimensions `B` and `H`). Along the last axis, it returns the sum of all entries strictly before each position:

$$
\text{out}[\dots, i] = \sum_{j < i} x[\dots, j] , \qquad \text{out}[\dots, 0] = 0 .
$$

**`first_true(mask)`** takes a 2D boolean tensor `mask` with `B` rows (batch size) and `T` columns (sequence length), and returns the index of the first `True` entry in each row:

$$
\text{out}[b] = \min\{\, t : \text{mask}[b, t] \,\} .
$$

If row `b` has no `True` entry, `out[b]` is `T`.

**`through_first_eos(ids, eos)`** takes a 2D tensor `ids` of token IDs and an end-of-sequence token ID `eos`. It returns a boolean mask marking every position up to and including the first `eos` in each row:

$$
\text{out}[b, t] = \big(\text{no } j < t \text{ has } \text{ids}[b, j] = \text{eos}\big) .
$$

The first `eos` of a row is `True`, and every position after it is `False` (including any later `eos`). A row with no `eos` is all `True`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(n,)`, `(B, n)`, or `(B, H, n)` | float32 |
| result of `exclusive_cumsum` | same shape as `x` | float32 |
| `mask` | `(B, T)` | bool |
| result of `first_true` | `(B,)` | integer |
| `ids` | `(B, T)` | integer |
| result of `through_first_eos` | `(B, T)` | bool |

`first_true` may return any integer type (your framework's default integer type is expected). Input `ids` uses your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX), and `eos` is a Python `int`.

## Tools

- `cumsum(x, axis)` (`torch.cumsum(x, dim=...)`, `jnp.cumsum`, `mx.cumsum`) computes the **inclusive** running sum along one axis in the same shape as `x`, where `out[i] = x[0] + ... + x[i]`.
- Taking `cumsum` of a boolean mask produces an integer running count of how many `True` entries have appeared so far.
- Comparing a running count against a threshold produces a new mask, and summing a mask along an axis counts the matching entries per row. Chaining these steps replaces a loop that stops at the first match.

> [!WARNING]
> `argmax` on a mask cannot distinguish a row whose first entry is `True` from a row with no `True` entry, because both return `0`. JAX and MLX accept boolean tensors in `argmax` directly. PyTorch raises `NotImplementedError` on boolean input, and casting the mask to integers still returns `0` in both cases.

## Example

```python
x = [[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -10.0, 5.0]]
exclusive_cumsum(x)
# [[0.0,  1.0,  3.0, 6.0],   # the inclusive sum would be [1.0, 3.0, 6.0, 10.0]
#  [0.0, 10.0, 10.0, 0.0]]

mask = [
    [False, True, False, True, False],
    [False, False, False, False, False],
    [True, True, False, False, False],
    [False, False, False, False, True],
]
first_true(mask)
# [1, 5, 0, 4]               # row 1 has no True entry, so it returns T = 5

ids = [[5, 2, 7, 2, 9], [1, 3, 4, 6, 8], [2, 2, 5, 2, 5]]
through_first_eos(ids, eos=2)
# [[ True,  True, False, False, False],  # the first 2 is kept, the second is not
#  [ True,  True,  True,  True,  True],  # no eos in this row
#  [ True, False, False, False, False]]  # eos at position 0
```

## Constraints

- `first_true` and `through_first_eos` are compared with the reference exactly.
- `exclusive_cumsum` is compared with a relative tolerance of `1e-4` and an absolute tolerance of `1e-4`.
- Every size (`B`, `H`, `n`, `T`) is between 1 and 16. Any size can be 1, and any two can be equal.
- Entries of `x` lie in $[-10, 10]$. Token IDs and `eos` lie in $[0, 1000]$.
- No Python loops are needed.

## Why this matters

Nucleus (top-p) filtering keeps each token while the probability mass ranked strictly before it is below `p`. That mass is the `exclusive_cumsum` of the sorted probabilities. Using the inclusive sum instead drops the token that crosses `p`. Inverse-CDF sampling finds the first cumulative sum above a threshold by counting how many entries are still below it, using the same pattern as `first_true`.

Speculative decoding uses a running product in the same way to accept draft tokens up to the first mismatch. In a batched decode loop, sequences finish at different steps, so rows keep holding tokens generated after their own `eos`. `through_first_eos` produces the mask that removes those trailing tokens.
