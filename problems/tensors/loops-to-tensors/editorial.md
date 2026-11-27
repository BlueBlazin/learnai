## The idea

Each function is a loop whose lines map onto operations from earlier in the module.

**`segment_means`.** The loop is `sums[seg[n]] += x[n]; counts[seg[n]] += 1`. Row `n`
goes to an output row chosen by data, and several rows go to the same one: an adding
scatter along axis 0, once with the rows and once with ones. Then
`sums / counts`. An empty segment has sum 0 and count 0, so replacing its count by 1
gives the zeros that the specification asks for, with no `where` afterwards.

**`common_prefix_len`.** The loop keeps one flag, "everything so far has matched",
and stops when it turns false. The flag at every position is the running product of
the 0/1 matches:

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `a == b` | 1 | 1 | 0 | 1 |
| `cumprod` | 1 | 1 | 0 | 0 |

A product cannot recover from a 0, which is the whole rule about matches after a
mismatch. The sum of the running product is the prefix length, `T` for equal rows.

**`repeated_bigram_ban`.** The loop is

```python
for t in range(T - 1):
    if row[t] == row[-1]:
        banned[row[t + 1]] = True
```

`range(T - 1)` is the slice `ids[:, :-1]`, and `row[t + 1]` over that range is
`ids[:, 1:]`: two views of the same tensor, one position apart, so that entry `t` of
one sits over its successor in the other. The comparison with the last token is a
mask of shape `(B, T - 1)`, and the write is a scatter into `(B, V)` at the columns
`ids[:, 1:]`. Using the adding form and counting, a token is banned when its count is
positive, and positions that do not match add 0.

The two slices also settle the edge rules. The final position is never a start,
because `ids[:, :-1]` does not contain it. The pair ending at the final position is
included, because `ids[:, 1:]` does contain it. With `T = 1` both slices are empty
and the counts stay 0.

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

In JAX and MLX the scatters are `zeros.at[seg].add(x)` and
`zeros.at[rows, second].add(starts_with_last)` with `rows = arange(B)[:, None]`.

Other correct forms, all of which passed the speed cases: a one-hot matrix
`(seg[:, None] == arange(S))` multiplied into `x`, which costs $O(N\,S\,D)$ and is
fine for a few segments; `torch.bincount(seg, minlength=S)` for the counts; the prefix
length as `first_true(a != b)` from Cumulative Sums as Control Flow; the ban as a
setting scatter of `True` at `where(match, second, V)` into `V + 1` columns, with the
spare column dropped, or as `((second[:, :, None] == arange(V)) & match[:, :, None])
.any(dim=1)`, which builds a `(B, T, V)` tensor and is the slowest of them.

## What the loop costs

Measured against the references above, at the sizes of the speed cases (in JAX,
1024 rows for `segment_means` and 512 for the ban):

| | PyTorch | JAX | MLX |
|---|---|---|---|
| `segment_means`, loop over the rows | more than 75x | more than 500x | 3x to 10x |
| `common_prefix_len`, loop over the rows | more than 100x | more than 1000x | more than 100x |
| `repeated_bigram_ban`, loop over the rows | more than 150x | more than 250x | more than 140x |

A tensor operation has a fixed cost that does not depend on how small its input is.
A loop pays it per row. The vectorized form pays it once, and the rest of its time
grows with the number of entries. JAX without `jit` is the extreme case: its cost per operation is the
highest of the three, and an `.at[...].add` inside a loop also copies the array it
updates.

The MLX entry for `segment_means` is the exception, and the reason is the reference:
MLX's adding scatter with an index of 2048 or 4096 rows took 5 to 17 times as long
as a one-hot matrix product with 4 or 8 segments, so a lazy row loop came within a
factor of 3 to 10 of it (measured at 2048 and 4096 rows). That speed case therefore
runs in PyTorch and JAX only.

A loop over a short axis is a different thing. Sixteen iterations over the positions,
each working on all 2048 rows, passed the speed cases in PyTorch and MLX.

## Common mistakes

- **The mean over all rows.** `sums / N` divides every segment by the total number
  of rows.
- **NaN for an empty segment.** `sums / counts` is `0 / 0` there. Clamp the count to
  at least 1 before dividing.
- **An assignment where an adding scatter is needed.** `sums[seg] = x` keeps one row
  per segment, and `sums[seg] += x` does the same in PyTorch and MLX: for the
  example it returned `[9, 90]` for segment 2, the last of its three rows, where the
  sum is `[15, 150]`.
- **Counting every match.** `(a == b).sum(dim=1)` includes the matches after the
  first mismatch.
- **`argmax` of the mismatch mask.** It is the prefix length when there is a
  mismatch, and 0 for equal rows, where `T` is right.
- **A bare `squeeze()`.** A sum with `keepdim=True` followed by `squeeze()` returns a
  tensor with no axes for a batch of one.
- **The successor by `roll`.** `torch.roll(ids, -1, dims=1)` gives the final position
  a successor: the first token of the row. The final position always matches the
  last token, so the first token of every row is banned.
- **Banning the matched token.** Scattering at `ids` where `ids == last`, instead of
  at the successors, bans the last token of every row and nothing else.
- **Dropping the last pair.** `ids[:, :-2]` against `ids[:, 1:-1]` stops one pair
  early, so a row that ends `..., 4, 4` does not ban 4.
- **A setting scatter of booleans.** `banned.scatter_(1, second, match)` writes a
  `False` for every pair that does not start with the last token. When a token
  follows the last token in one place and another token elsewhere, both writes land
  on its column and one is lost. Which one is not specified for repeated indices; on
  this machine all three frameworks kept the later write.
- **The row loop.** Correct, and rejected by the speed cases.

## Complexity

`segment_means` is $O(N\,D + S\,D)$, `common_prefix_len` $O(B\,T)$, and
`repeated_bigram_ban` $O(B\,T + B\,V)$. The one-hot forms cost $O(N\,S\,D)$ and
$O(B\,T\,V)$, which is the difference between a ban over 16 tokens and one over a
real vocabulary.

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

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The frameworks agree on the trap of this problem: an indexed `+=` is a read, an add,
and a write, and with repeated indices the writes overwrite each other. Only the
named adding operations accumulate. They differ in what those are called and in their
speed: `index_add_` and `.at[].add` take the row index as it is, `scatter_add_` wants
it expanded to the shape of the source as in Gathering Along an Axis, and in MLX the
adding scatter was slow enough at 4096 rows that a matrix product with a one-hot
matrix beat it by a wide margin.
