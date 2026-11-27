## The idea

A loop that scans a row keeps a running variable and branches on it. A cumulative sum
is the whole history of that variable, one value per position, so the branch becomes
a comparison on a tensor.

**`exclusive_cumsum`.** `cumsum` is inclusive: entry `i` is the total after adding
`x[i]`. The total before adding it is the previous inclusive sum, so the answer is
the inclusive sums shifted one place to the right: a 0 in front, the last sum dropped.

**`first_true`.** The running count of true entries, for `. # . # .`:

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `mask` | F | T | F | T | F |
| `cumsum(mask)` | 0 | 1 | 1 | 2 | 2 |
| `cumsum == 0` | T | F | F | F | F |

The count is 0 on exactly the positions before the first true entry, and there are
as many of those as the index of that entry. A row with no true entry has a count of
0 everywhere, and the same sum returns `T` with no special case.

**`through_first_eos`.** Position `t` is kept when no `eos` comes strictly before it:
the exclusive count of `eos` tokens is 0 there. The first `eos` has none before it,
so it is kept; every later position has at least one.

## Reference solution (PyTorch)

```python
def exclusive_cumsum(x):
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask):
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids, eos):
    is_eos = (ids == eos).to(torch.int64)
    return exclusive_cumsum(is_eos) == 0
```

The JAX version is the same with `jnp.concatenate` and `axis=`. MLX has the exclusive
sum built in: `mx.cumsum(x, axis=-1, inclusive=False)`.

`cumsum(x) - x` is the other common form of the exclusive sum, and Nucleus (Top-p)
Filtering uses it. It is exact for integers. In float32 it subtracts after rounding:
for `x = [0.1, 1e8, 0.3]` it returned `[0, 0, 1e8]`, where the shifted form returned
`[0, 0.1, 1e8]`. And where `x` holds `inf` it returns `inf - inf`, which is NaN.

`through_first_eos` can also be composed from the other two pieces of this module:
`arange(T)[None, :] <= first_true(ids == eos)[:, None]`.

## Common mistakes

- **The inclusive sum.** For "how much came before", `cumsum` alone is off by the
  current entry. In a top-p filter that drops the token that crosses the threshold.
- **Shifting with `roll`.** `roll` wraps, so position 0 receives the last inclusive
  sum, the total of the row, where a 0 belongs.
- **`argmax` for the first true entry.** It is right whenever the row has a true
  entry and returns 0 when it has none, which reads as "found at position 0".
- **Counting false entries.** `(~mask).sum()` counts the false entries after the
  first true one as well.
- **A bare `squeeze()` on the result.** A sum with `keepdim=True` followed by
  `squeeze()` returns a tensor with no axes for a batch of one. Sum without
  `keepdim`, which removes exactly the position axis.
- **Dropping the `eos`.** `cumsum(is_eos) == 0` is false at the first `eos` itself.
  The question is whether an `eos` came *before* this position.
- **`ids != eos`.** It is false at every `eos` and true everywhere else, including
  after the first one.
- **`cumsum(is_eos) <= 1`.** It keeps the first `eos`, and then everything up to the
  second.
- **The wrong axis.** `cumsum` along axis 0 runs down the batch and returns the same
  shape.

## Complexity

One pass along the axis for each function: $O(B\,T)$ time and memory.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| running sum | `torch.cumsum(x, dim=d)` | `jnp.cumsum(x, axis=d)` | `mx.cumsum(x, axis=d)` |
| no axis given | `TypeError` | flattens first | flattens first |
| exclusive form | none | `jnp.cumulative_sum(x, axis=d, include_initial=True)`, then drop the last entry | `mx.cumsum(x, axis=d, inclusive=False)` |
| from the end | flip, `cumsum`, flip | flip, `cumsum`, flip; `jax.lax.cumsum(x, reverse=True)` | `mx.cumsum(x, axis=d, reverse=True)` |
| `cumsum` of a boolean tensor | int64 | int32 | int32 |
| integer count minus a boolean tensor | `NotImplementedError`: cast the mask first | int32 | int32 |
| `argmax` of a boolean tensor | `NotImplementedError`: cast first | int32 indices | uint32 indices |
| `argmax` of a row of equal values | 0 | 0 | 0 |
| running max | `torch.cummax(m, dim=d).values`, also for bool | `jax.lax.cummax(m, axis=d)`; `TypeError` for bool | `mx.cummax(m, axis=d)`, also for bool |
| `cumprod` of a boolean tensor | int64 | int32 | bool |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

MLX is the only one of the three with an exclusive and a reversed running sum as
arguments. PyTorch is the strictest about booleans: `cumsum` accepts a mask, but
`argmax` and subtraction do not, so the cast to an integer type is written out there
and can be left out in JAX and MLX. In all three a running count of booleans is an
integer tensor of the default width, which is what the comparisons in this problem
need.
