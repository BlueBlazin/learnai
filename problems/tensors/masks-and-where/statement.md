Implement three functions that use a boolean mask to select which entries of a tensor count. Masked-out entries can hold any float32 value, including `inf`, `-inf`, and `NaN`, and none of those values may affect the result.

## Specification

```python
def fill_masked(scores, keep): ...
def masked_mean(x, mask): ...
def masked_max(x, mask): ...
```

**`fill_masked(scores, keep)`** takes a 2D tensor `scores` with `B` rows and `V` columns, along with a boolean mask `keep` of the same shape. It returns a tensor of shape `(B, V)` that keeps `scores[b, v]` where `keep[b, v]` is true and replaces every other entry with $-\infty$:

$$
\text{out}[b, v] = \begin{cases} \text{scores}[b, v] & \text{if keep}[b, v], \\ -\infty & \text{otherwise.} \end{cases}
$$

The fill value must be exact `-inf`. A large finite number such as `-1e9` is not accepted.

**`masked_mean(x, mask)`** takes a 3D tensor `x` of shape `(B, T, D)` representing `B` rows of `T` steps with `D` features each. Its boolean `mask` has shape `(B, T)`, where `mask[b, t]` is true when step `t` of row `b` is kept. Let $n_b$ be the number of kept steps in row `b`. The function returns the mean over the kept steps for each row and feature, with shape `(B, D)`:

$$
\text{out}[b, d] = \frac{1}{n_b} \sum_{t\,:\,\text{mask}[b, t]} x[b, t, d] .
$$

If a row keeps no steps ($n_b = 0$), its output is all zeros.

**`masked_max(x, mask)`** takes the same `x` and `mask` arguments and returns the largest kept value for each row and feature, with shape `(B, D)`:

$$
\text{out}[b, d] = \max_{t\,:\,\text{mask}[b, t]} x[b, t, d] .
$$

If a row keeps no steps, its output is $-\infty$ in every feature.

Kept entries in `scores` and `x` are always finite, while masked-out entries may be any float32 value (including `inf`, `-inf`, and `NaN`). Output shapes must remain `(B, V)` and `(B, D)` even when `B`, `T`, `V`, or `D` is 1.

## Shapes

| name | shape | dtype |
|---|---|---|
| `scores` | `(B, V)` | float32 |
| `keep` | `(B, V)` | bool |
| result of `fill_masked` | `(B, V)` | float32 |
| `x` | `(B, T, D)` | float32 |
| `mask` | `(B, T)` | bool |
| result of `masked_mean`, `masked_max` | `(B, D)` | float32 |

## Tools

- `where(condition, a, b)` (`torch.where`, `jnp.where`, `mx.where`) selects `a` where `condition` is true and `b` elsewhere, element by element. The three arguments broadcast against each other, and `a` or `b` can be a Python scalar. No arithmetic runs on the unchosen entry.
- A mask with fewer axes than the data aligns from the right, just like any other broadcast operand. As in Broadcasting on Purpose, indexing with `None` inserts a size-1 axis where needed.
- To reduce over kept entries only, first replace masked-out entries with an identity value that cannot change the reduction. That value depends on the reduction.
- `mask.sum(...)` counts true entries and returns an integer tensor.

> [!WARNING]
> Multiplying by a mask is not selection because `0 * inf` and `0 * nan` produce `NaN`. In PyTorch and MLX, `x * mask` produces `NaN` wherever a masked-out entry is non-finite, which then poisons a sum or max. JAX does the same when the mask broadcasts against `x`, and returns `0` only when a boolean mask has the exact shape of `x`.

Boolean indexing such as `x[mask]` returns the kept entries as a flat 1D tensor whose length depends on the data, losing the row structure. MLX does not support it at all. None of these functions needs boolean indexing.

## Example

```python
scores = [[1.0, 2.0, inf, 4.0], [nan, -1.0, 0.5, -inf]]
keep = [[True, True, False, False], [False, True, True, False]]
fill_masked(scores, keep)
# [[ 1.0,  2.0, -inf, -inf],
#  [-inf, -1.0,  0.5, -inf]]

x = [
    [[0.0, 1.0], [inf, nan], [20.0, 21.0], [30.0, -inf]],
    [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0], [130.0, 131.0]],
    [[200.0, 201.0], [nan, 211.0], [220.0, -inf], [230.0, 231.0]],
]  # shape (3, 4, 2); finite entries are 100*b + 10*t + d
mask = [
    [True, False, True, False],  # steps 0 and 2
    [False, False, False, True],  # step 3
    [False, False, False, False],  # nothing
]
masked_mean(x, mask)
# [[ 10.0,  11.0],                   # (0 + 20) / 2 and (1 + 21) / 2
#  [130.0, 131.0],                   # one step: divided by 1, not by T = 4
#  [  0.0,   0.0]]                   # no step kept
masked_max(x, mask)
# [[ 20.0,  21.0],
#  [130.0, 131.0],
#  [ -inf,  -inf]]                   # no step kept
```

Row 0 drops the step `[inf, nan]`. Multiplying by a float mask turns that step into `[nan, nan]`, which makes the row's mean and max `NaN`.

## Constraints

- `fill_masked` and `masked_max` copy values directly and are checked for exact equality. `masked_mean` is checked with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`.
- Each dimension size (`B`, `T`, `V`, `D`) is between 1 and 16. Any size can be 1, and any two sizes can be equal.
- Kept entries lie in $[-1000, 1000]$.
- No Python loops are needed.

## Why this matters

Causal and Padding Masks applies `fill_masked` to attention scores, where a finite stand-in for `-inf` leaks probability mass into masked positions. Sequence Log-Likelihood and Perplexity sums log-probabilities over real tokens where masked positions can hold `-inf`, so it must select with `where`.

Pooling and Normalizing Embeddings averages hidden states over real tokens, dividing by each sequence's real length rather than the padded length. In serving systems, masked positions often hold uninitialized padding or logits already set to `-inf` by an earlier filter. A reduction that multiplies by a mask appears to work during testing and silently returns `NaN` once a batch contains one of those values.
