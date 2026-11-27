Write three functions that use a boolean mask to decide which entries of a tensor
count. The entries that do not count can hold anything, including `inf`, `-inf`, and
NaN, and none of it may reach the result.

## Specification

```python
def fill_masked(scores, keep): ...
def masked_mean(x, mask): ...
def masked_max(x, mask): ...
```

**`fill_masked(scores, keep)`** takes `scores` of shape `(B, V)` and a boolean `keep`
of the same shape, and returns `scores` with $-\infty$ at every entry that is not
kept:

$$
\text{out}[b, v] = \begin{cases} \text{scores}[b, v] & \text{if keep}[b, v], \\ -\infty & \text{otherwise.} \end{cases}
$$

The fill is exactly `-inf`. A large finite number such as `-1e9` is not accepted.

**`masked_mean(x, mask)`** takes `x` of shape `(B, T, D)` and a boolean `mask` of
shape `(B, T)`, where `mask[b, t]` says whether step `t` of row `b` is kept. It
returns the mean of the kept steps of every row, shape `(B, D)`. With
$n_b$ the number of kept steps of row `b`:

$$
\text{out}[b, d] = \frac{1}{n_b} \sum_{t\,:\,\text{mask}[b, t]} x[b, t, d] .
$$

A row that keeps no step ($n_b = 0$) returns zeros.

**`masked_max(x, mask)`** takes the same arguments and returns the largest kept value
of every feature, shape `(B, D)`:

$$
\text{out}[b, d] = \max_{t\,:\,\text{mask}[b, t]} x[b, t, d] .
$$

A row that keeps no step returns $-\infty$ in every feature.

- `scores`, `x`: float32 tensors. An entry that is kept is finite. An entry that is
  not kept may be any float32 value, including `inf`, `-inf`, and NaN, and must not
  affect the result.
- `keep`, `mask`: boolean tensors.
- Every function returns a float32 tensor. The result shapes are `(B, V)` and
  `(B, D)` for every input, also when `B`, `T`, `V`, or `D` is 1.

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

- `where(condition, a, b)` (`torch.where`, `jnp.where`, `mx.where`) picks `a` where
  the condition is true and `b` elsewhere, entry by entry. The three arguments are
  broadcast against each other, and `a` or `b` can be a Python number. No arithmetic
  is done on the entry that is not picked.
- A mask with fewer axes than the data lines up from the right, like any other
  operand. As in Broadcasting on Purpose, `None` puts the missing axis where it
  belongs.
- To reduce over the kept entries only, first replace the others with a value that
  cannot change the reduction. Which value that is depends on the reduction.
- `mask.sum(...)` counts true entries and returns integers.

> [!WARNING]
> Multiplying by a mask is not selecting. `0 * inf` and `0 * nan` are NaN. In
> PyTorch and MLX, `x * mask` puts a NaN wherever a dropped entry was not finite, and
> a sum or a max over it is NaN too. JAX does the same when the mask is broadcast
> against `x`, and returns 0 there when a boolean mask has the shape of `x`.

Reading with a boolean index, `x[mask]`, returns the kept entries as one flat list
whose length depends on the data, so the rows are no longer apart. MLX does not
support it at all. None of these functions needs it.

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
]  # shape (3, 4, 2); the finite entries are 100 b + 10 t + d
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

Row 0 drops the step `[inf, nan]`. Multiplied by a float mask it becomes
`[nan, nan]`, and the row's mean and max are NaN.

## Constraints

- `fill_masked` and `masked_max` copy values and are compared with the reference
  exactly. `masked_mean` is compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-6`.
- Every size is between 1 and 16. Any of them can be 1, and any two can be equal.
- Kept entries lie in $[-1000, 1000]$.
- No Python loops are needed.

## Why this matters

Causal and Padding Masks is `fill_masked` on attention scores, and it also refuses a
finite stand-in for `-inf`. Sequence Log-Likelihood and Perplexity sums
log-probabilities over the real positions of each row, where an uncounted entry can be
`-inf`, so it has to select with `where`. Pooling and Normalizing Embeddings takes
the mean over the real tokens of each sequence, divided by their number and not by
the padded length; its hidden states are finite, which is the one situation where a
multiplication by the mask is safe. In a server the entries behind a mask are padding, or logits
that an earlier filter set to `-inf`: a masked reduction that multiplies gives the
right answer until a batch holds one of them.
