Write three functions that are easy as a Python loop over rows, and write them
without that loop: a mean per segment, the length of the common prefix of two
sequences, and the tokens that would repeat a bigram. Each one is a combination of
earlier drills of this module, and the work is seeing which.

## Specification

```python
def segment_means(x, seg, S): ...
def common_prefix_len(a, b): ...
def repeated_bigram_ban(ids, V): ...
```

**`segment_means(x, seg, S)`** takes `x` of shape `(N, D)` and segment ids `seg` of
shape `(N,)` with $0 \le \text{seg}[n] < S$, and returns the mean of the rows of
every segment, shape `(S, D)`:

$$
\text{out}[s] = \frac{1}{|R_s|} \sum_{n \in R_s} x[n] , \qquad R_s = \{\, n : \text{seg}[n] = s \,\} ,
$$

and a row of zeros for a segment with no rows. Segment ids come in any order.

**`common_prefix_len(a, b)`** takes two id tensors of shape `(B, T)` and returns, for
every row, the number of leading positions at which the two rows agree, shape `(B,)`:

$$
\text{out}[i] = \max\{\, n \le T : a[i, t] = b[i, t] \text{ for all } t < n \,\} .
$$

It is `T` for equal rows and 0 when the first entries differ. Matches after the first
mismatch do not count.

**`repeated_bigram_ban(ids, V)`** takes token ids of shape `(B, T)` with
$0 \le \text{ids}[b, t] < V$ and returns a boolean tensor of shape `(B, V)`. Let
$\ell_b = \text{ids}[b, T - 1]$ be the last token of row $b$. Token $v$ is banned when
appending it would create a pair $(\ell_b, v)$ that the row already contains:

$$
\text{out}[b, v] = \big(\text{some } t \text{ with } 0 \le t \le T - 2 \text{ has } \text{ids}[b, t] = \ell_b \text{ and } \text{ids}[b, t + 1] = v\big) .
$$

- A pair starts at every position except the final one. The final position holds
  $\ell_b$ and has no successor in the row, so it bans nothing.
- The pair that ends at the final position, $(\text{ids}[b, T - 2],\ \ell_b)$, counts
  like any other: a row ending in the same token twice bans that token.
- A row with $T = 1$ contains no pair, and nothing is banned.

**No Python loop over rows or elements**: not over the `N` rows of `x` and not over
the `B` rows of a batch. The speed cases rule those out. A loop over a short axis (the
`T` positions, the `S` segments) may pass them and is not the point of the exercise.

- `x`: a float32 tensor; `segment_means` returns float32.
- `seg`, `a`, `b`, `ids`: integer tensors of your framework's default integer type
  (int64 in PyTorch, int32 in JAX and MLX). `S`, `V`: positive Python ints.
- `common_prefix_len` returns an integer tensor; any integer width is accepted.
  `repeated_bigram_ban` returns a boolean tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `seg` | `(N,)` | integer |
| result of `segment_means` | `(S, D)` | float32 |
| `a`, `b` | `(B, T)` | integer |
| result of `common_prefix_len` | `(B,)` | integer |
| `ids` | `(B, T)` | integer |
| result of `repeated_bigram_ban` | `(B, V)` | bool |

## Tools

Nothing new is needed. The pieces are:

- the adding scatter of Scatter and Indexed Writes, which sends many inputs to one
  output entry and accumulates them (`index_add_` and `scatter_add_` in PyTorch,
  `.at[...].add` in JAX and MLX);
- a running sum or product along an axis, from Cumulative Sums as Control Flow, which
  turns "until the first time something happens" into arithmetic;
- comparisons of a tensor with a shifted slice of itself, such as `ids[:, :-1]` with
  `ids[:, 1:]`, which line up every position with its successor.

To find the tensor form of a loop, write the loop first, and ask of each line whether
it reads or writes one entry per iteration (an index, a gather, a scatter), keeps a
running variable (a cumulative operation), or branches (a mask and `where`).

> [!WARNING]
> In PyTorch and MLX, `out[idx] += values` with a repeated index does not
> accumulate: each repeated entry receives one of the values. Accumulating needs the
> adding scatter.

## Example

```python
x = [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]  # shape (4, 2)
segment_means(x, seg=[2, 0, 2, 2], S=4)
# [[3.0, 30.0],       # segment 0: row 1
#  [0.0,  0.0],       # segment 1: empty
#  [5.0, 50.0],       # segment 2: rows 0, 2, 3
#  [0.0,  0.0]]       # segment 3: empty

a = [[5, 6, 7, 8], [5, 6, 7, 8], [1, 2, 3, 4], [9, 9, 9, 9]]
b = [[5, 6, 0, 8], [5, 6, 7, 8], [0, 2, 3, 4], [9, 9, 9, 0]]
common_prefix_len(a, b)
# [2, 4, 0, 3]        # row 0 agrees again at position 3, which does not count

ids = [
    [1, 2, 3, 1, 4, 1],   # last token 1; earlier pairs (1, 2) and (1, 4)
    [3, 3, 0, 1, 2, 3],   # last token 3; earlier pairs (3, 3) and (3, 0)
    [0, 1, 2, 3, 4, 5],   # last token 5 occurs nowhere else
    [2, 2, 2, 2, 2, 2],   # last token 2; the pair (2, 2)
]
repeated_bigram_ban(ids, V=6)
# [[False, False,  True, False,  True, False],    # 2 and 4
#  [ True, False, False,  True, False, False],    # 0 and 3
#  [False, False, False, False, False, False],
#  [False, False,  True, False, False, False]]    # 2
```

## Constraints

- `common_prefix_len` and `repeated_bigram_ban` are compared with the reference
  exactly. `segment_means` is compared to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`.
- Apart from the speed sizes below: $1 \le N \le 64$, and `D`, `S`, `B`, `T`, `V`
  are between 1 and 16. Any size can be 1, and any two can be equal. Entries of `x` lie in $[-2, 2]$ in
  the random cases and are whole numbers up to 90 otherwise. Token ids lie in
  $[0, 1000)$ for `common_prefix_len`.
- **Speed.** Each function is timed against the reference written in the same
  framework, in the same process (interleaved runs, median of 3 after a warm-up
  call), and must take at most **20 times** as long. Its result at that size is
  also checked. In JAX two of the cases use a quarter of the rows.
  - `segment_means` with $N = 4096$ (1024 in JAX), $D = 16$, $S = 8$, in PyTorch and
    JAX. This case does not run in MLX, where the adding scatter is itself slow and
    a row loop measured only 3 to 10 times the reference.
  - `common_prefix_len` with $B = 2048$, $T = 16$.
  - `repeated_bigram_ban` with $B = 2048$ (512 in JAX), $T = 16$, $V = 16$.

  A loop over the rows measured more than 75 times the reference in every case and
  framework where the case runs. Of the loop-free solutions tried, about forty
  formulations across the three frameworks, the slowest measured 12 times (the ban
  through paired indexing and `torch.unique` in PyTorch) and most were under 5 times.

## Why this matters

`common_prefix_len` is the acceptance count of speculative decoding: Verifying a
Greedy Draft takes the running product of the matches for one sequence, and Ragged
Acceptance in a Batch does it for every row at once. Frequency and Presence Penalties
counts tokens with the adding scatter of `segment_means`, and Sparse Expert Dispatch
adds each expert's outputs back to its tokens with `index_add`. `repeated_bigram_ban`
is the no-repeat rule for n-grams of size 2, the kind of logit processor that
Vectorized Repetition Penalty and DRY Repetition Penalty are. These run for every
sequence at every decoding step, so a loop over the batch costs a fixed Python
overhead per row per token, and that is what the speed cases measure.
