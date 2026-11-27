## The idea

Each function starts as a simple loop whose lines map directly to vectorized tensor operations.

**`segment_means`.** The row loop accumulates `sums[seg[n]] += x[n]` and `counts[seg[n]] += 1`. Because multiple rows can map to the same output segment `seg[n]`, both updates are adding scatters along axis 0. Scatter the rows of `x` to get `sums`, and scatter a vector of ones to get `counts`. Then divide `sums / counts`. An empty segment has sum `0` and count `0`, so clamping `counts` to a minimum of `1` yields `0 / 1 = 0` without needing a separate `where`.

**`common_prefix_len`.** A loop tracks a single boolean flag ("all positions so far have matched") and stops on the first mismatch. That flag at every position is the cumulative product (`cumprod`) of the 0/1 match indicator `a == b`:

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `a == b` | 1 | 1 | 0 | 1 |
| `cumprod` | 1 | 1 | 0 | 0 |

Once a `0` appears, the running product stays `0` for all later positions, which ignores matches after the first mismatch. Summing the running product along the sequence axis gives the prefix length, returning `T` when all entries match.

**`repeated_bigram_ban`.** For a single row, the loop is:

```python
for t in range(T - 1):
    if row[t] == row[-1]:
        banned[row[t + 1]] = True
```

The range `0` to `T - 2` corresponds to the slice `ids[:, :-1]`, and the successor `row[t + 1]` corresponds to `ids[:, 1:]`. Both slices have shape `(B, T - 1)`, aligning every token with the token that follows it. Comparing `ids[:, :-1]` against the last token `ids[:, -1:]` gives a 0/1 match mask of shape `(B, T - 1)`. Scatter-add that mask into a `(B, V)` count buffer at column indices `ids[:, 1:]`. Non-matching positions add `0`, and any token with a positive final count is banned.

These two slices also handle all edge cases automatically. The final position never starts a bigram because `ids[:, :-1]` excludes it. The bigram ending at the final position is included because `ids[:, 1:]` contains it. When `T = 1`, both slices are empty and all counts remain `0`.

## Reference solution (PyTorch)

```python
def segment_means(x, seg, S):
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype).index_add_(0, seg, x)
    counts = torch.zeros(S, dtype=x.dtype).index_add_(0, seg, torch.ones_like(x[:, 0]))
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a, b):
    still_equal = torch.cumprod((a == b).to(torch.int64), dim=1)
    return still_equal.sum(dim=1)


def repeated_bigram_ban(ids, V):
    first = ids[:, :-1]
    second = ids[:, 1:]
    starts_with_last = (first == ids[:, -1:]).to(torch.int64)
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    counts.scatter_add_(1, second, starts_with_last)
    return counts > 0
```

In JAX and MLX, the scatters are `zeros.at[seg].add(x)` and `zeros.at[rows, second].add(starts_with_last)`, where `rows = arange(B)[:, None]`.

Several other vectorized formulations also pass the speed tests:

- Multiplying a one-hot matrix `(seg[:, None] == arange(S))` with `x` for `segment_means`. This costs $O(N S D)$ and works well when `S` is small.
- Using `torch.bincount(seg, minlength=S)` for the segment counts.
- Computing `common_prefix_len` with `first_true(a != b)` from Cumulative Sums as Control Flow.
- Implementing `repeated_bigram_ban` with a setting scatter of `True` into `V + 1` columns using `where(match, second, V)`, where `match = (first == ids[:, -1:])`, and dropping the extra column.
- Computing `((second[:, :, None] == arange(V)) & match[:, :, None]).any(dim=1)` for `repeated_bigram_ban`. This allocates a `(B, T, V)` tensor and is the slowest of these alternatives.

## What the loop costs

Measured against the reference solutions above at the speed-test sizes (in JAX, `N = 1024` for `segment_means` and `B = 512` for `repeated_bigram_ban`):

| | PyTorch | JAX | MLX |
|---|---|---|---|
| `segment_means`, loop over the rows | more than 75x | more than 500x | 3x to 10x |
| `common_prefix_len`, loop over the rows | more than 100x | more than 1000x | more than 100x |
| `repeated_bigram_ban`, loop over the rows | more than 150x | more than 250x | more than 140x |

Every tensor operation has a fixed dispatch cost regardless of input size. A row loop pays that overhead on every row, whereas a vectorized call pays it once. Un-jitted JAX has the highest per-op dispatch cost of the three frameworks, and calling `.at[...].add` inside a loop also copies the array on every iteration.

The MLX result for `segment_means` is an exception because MLX's adding scatter is slow at 2048 to 4096 rows. It took 5x to 17x longer than a one-hot matrix product with 4 to 8 segments, bringing a lazy row loop within 3x to 10x of the reference. For that reason, the `segment_means` speed test runs only in PyTorch and JAX.

Looping over a short axis behaves differently. Iterating 16 times over `T` while operating on all 2048 rows at once still passed the speed tests in PyTorch and MLX.

## Common mistakes

- **Dividing by the total row count.** Computing `sums / N` divides every segment by the total number of rows instead of its own segment count.
- **NaNs on empty segments.** Dividing `sums / counts` produces `0 / 0 = NaN` when a segment has no rows. Clamp `counts` to at least `1` before dividing.
- **Using assignment or `+=` instead of an adding scatter.** `sums[seg] = x` keeps only one row per segment, and `sums[seg] += x` does the same in PyTorch and MLX. In the worked example, `sums[seg] += x` returns `[9, 90]` for segment 2 (the last of its three rows) instead of `[15, 150]`.
- **Counting all matches.** `(a == b).sum(dim=1)` counts matches that occur after the first mismatch.
- **Using `argmax` on the mismatch mask.** `argmax(a != b)` returns the prefix length when a mismatch exists, but returns `0` instead of `T` when the two rows are identical.
- **Calling bare `squeeze()`.** Summing with `keepdim=True` and then calling `squeeze()` drops the batch axis when `B = 1`, returning a 0D scalar instead of shape `(1,)`.
- **Using `roll` to get successors.** `torch.roll(ids, -1, dims=1)` wraps the first token around as the successor to the final token. Because the final token always matches itself, this wrongly bans the first token of every row.
- **Banning the matched token instead of its successor.** Scattering at `ids` where `ids == last` bans the last token of every row rather than the tokens that followed it.
- **Dropping the last bigram.** Comparing `ids[:, :-2]` against `ids[:, 1:-1]` stops one pair too early, so a row ending in `..., 4, 4` fails to ban `4`.
- **Using a setting scatter of booleans.** `banned.scatter_(1, second, match)` writes `False` for every non-matching pair. If a token follows the last token at one position and a different token elsewhere, both writes target the same column and one overwrites the other. Duplicate-index write order is unspecified, and all three frameworks kept the later write in testing.
- **Looping over rows in Python.** Correct numerically, but rejected by the speed tests.

## Complexity

`segment_means` runs in $O(N D + S D)$ time, `common_prefix_len` in $O(B T)$ time, and `repeated_bigram_ban` in $O(B T + B V)$ time. The one-hot alternatives cost $O(N S D)$ and $O(B T V)$, which becomes prohibitive for `repeated_bigram_ban` when `V` is a full LLM vocabulary rather than 16.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| add rows of `x` into rows `seg` | `out.index_add_(0, seg, x)` | `out.at[seg].add(x)` | `out.at[seg].add(x)` |
| the same, as a scatter | `out.scatter_add_(0, index, x)` with `seg` expanded to `(N, D)` | `jax.ops.segment_sum(x, seg, num_segments=S)` | none found |
| `scatter_add_` with an unexpanded `(N, 1)` index | writes column 0 only | not applicable | not applicable |
| count per segment | `torch.bincount(seg, minlength=S)` | `jnp.bincount(seg, length=S)` | no `mx.bincount`; `zeros.at[seg].add(1)` |
| `out[seg] += x` with repeated ids | no accumulation | `TypeError`: immutable | no accumulation |
| `out[seg] = x` with repeated ids | the last row written stays | `.at[seg].set(x)`: the last row stays | the last row stays |
| add along axis 1, an index per row | `out.scatter_add_(1, idx, src)` | `out.at[rows, idx].add(src)` | `out.at[rows, idx].add(src)` |
| a scatter that keeps the maximum | `out.scatter_reduce(1, idx, src, reduce="amax")` | `out.at[rows, idx].max(src)` | `out.at[rows, idx].maximum(src)` |
| an adding scatter with an empty index | no change | no change | no change |
| `cumprod` of a boolean tensor | int64 | int32 | bool; its `sum` is int32 |

Every entry was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

All three frameworks share the main pitfall of this problem. Indexed `+=` executes as a read, an add, and a write, so duplicate indices overwrite each other instead of accumulating. Only named adding scatters accumulate safely. The frameworks differ in their APIs and performance. `index_add_` and `.at[...].add` accept a 1D row index directly, while `scatter_add_` requires expanding the index to match the source shape. In MLX, the adding scatter is slow enough at 4096 rows that multiplying by a one-hot matrix is substantially faster.
