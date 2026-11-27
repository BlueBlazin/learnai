Implement three vectorized functions without looping over rows in Python. You will compute per-segment row means, the common prefix length of paired sequences, and a mask of tokens that would repeat an existing bigram.

## Specification

```python
def segment_means(x, seg, S): ...
def common_prefix_len(a, b): ...
def repeated_bigram_ban(ids, V): ...
```

**`segment_means(x, seg, S)`** takes a float32 tensor `x` of shape `(N, D)`, an integer tensor `seg` of shape `(N,)` with values $0 \le \text{seg}[n] < S$, and a positive integer `S` for the number of segments. It averages the rows of `x` assigned to each segment `s` and returns a float32 tensor of shape `(S, D)`:

$$
\text{out}[s] = \frac{1}{|R_s|} \sum_{n \in R_s} x[n], \qquad R_s = \{\, n : \text{seg}[n] = s \,\}
$$

If a segment has no rows (`|R_s| = 0`), its output row is all zeros. Segment IDs in `seg` can appear in any order.

**`common_prefix_len(a, b)`** takes two integer tensors `a` and `b` of shape `(B, T)`, where `B` is the batch size and `T` is the sequence length. For each row `i`, it returns the number of leading positions where `a[i]` and `b[i]` are equal, producing an integer tensor of shape `(B,)`:

$$
\text{out}[i] = \max\{\, n \le T : a[i, t] = b[i, t] \text{ for all } t < n \,\}
$$

The result is `T` for identical rows and `0` when the first entries differ. Matches after the first mismatch do not count.

**`repeated_bigram_ban(ids, V)`** takes an integer tensor `ids` of shape `(B, T)` with token IDs $0 \le \text{ids}[b, t] < V$, where `V` is the vocabulary size. It returns a boolean tensor of shape `(B, V)`. For each row `b`, let $\ell_b = \text{ids}[b, T - 1]$ be the last token. Token `v` is banned (`True`) if appending `v` to row `b` would create a bigram $(\ell_b, v)$ that already appears in row `b`:

$$
\text{out}[b, v] = \big(\text{there exists } t \text{ with } 0 \le t \le T - 2 \text{ such that } \text{ids}[b, t] = \ell_b \text{ and } \text{ids}[b, t + 1] = v\big)
$$

- A bigram starts at every position except the final one. The final position holds $\ell_b$ and has no successor, so it bans nothing on its own.
- The bigram ending at the final position, $(\text{ids}[b, T - 2], \ell_b)$, counts like any other. A row that ends with the same token twice bans that token.
- A row with `T = 1` contains no bigrams, so every entry in its output row is `False`.

**No Python loop over rows or elements.** Do not iterate over the `N` rows of `x` or the `B` rows of a batch. The speed tests reject row loops. Looping over a short axis (such as `T` or `S`) may pass the speed tests, though you should aim for a fully vectorized solution.

## Shapes

| Name | Shape | Dtype |
|---|---|---|
| `x` | `(N, D)` | `float32` |
| `seg` | `(N,)` | integer |
| `segment_means` output | `(S, D)` | `float32` |
| `a`, `b` | `(B, T)` | integer |
| `common_prefix_len` output | `(B,)` | integer |
| `ids` | `(B, T)` | integer |
| `repeated_bigram_ban` output | `(B, V)` | `bool` |

`S` and `V` are positive Python ints. Integer inputs use your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX). `common_prefix_len` may return any integer width.

## Tools

You can build all three functions from operations introduced earlier in this track:

- **Adding scatters** (from Scatter and Indexed Writes) accumulate multiple inputs into the same output entry (`index_add_` and `scatter_add_` in PyTorch, `.at[...].add` in JAX and MLX).
- **Cumulative operations** (from Cumulative Sums as Control Flow) turn "until the first time a condition fails" into elementwise arithmetic.
- **Shifted slices** such as `ids[:, :-1]` and `ids[:, 1:]` align every position with its immediate successor.

A reliable way to vectorize a loop is to write the loop first, then translate each line. Data-dependent reads and writes map to gathers or scatters, running state maps to cumulative operations, and branches map to masks and `where`.

> [!WARNING]
> In PyTorch and MLX, `out[idx] += values` does not accumulate when `idx` contains duplicates. Only one value is kept per repeated index. Use an adding scatter instead.

## Example

```python
x = [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]  # shape (4, 2)
segment_means(x, seg=[2, 0, 2, 2], S=4)
# [[3.0, 30.0],       # segment 0: row 1
#  [0.0,  0.0],       # segment 1: empty
#  [5.0, 50.0],       # segment 2: rows 0, 2, 3 -> mean of [1, 10], [5, 50], [9, 90]
#  [0.0,  0.0]]       # segment 3: empty

a = [[5, 6, 7, 8], [5, 6, 7, 8], [1, 2, 3, 4], [9, 9, 9, 9]]
b = [[5, 6, 0, 8], [5, 6, 7, 8], [0, 2, 3, 4], [9, 9, 9, 0]]
common_prefix_len(a, b)
# [2, 4, 0, 3]        # row 0 matches again at position 3, which does not count

ids = [
    [1, 2, 3, 1, 4, 1],   # last token 1; earlier pairs (1, 2) and (1, 4)
    [3, 3, 0, 1, 2, 3],   # last token 3; earlier pairs (3, 3) and (3, 0)
    [0, 1, 2, 3, 4, 5],   # last token 5 appears nowhere else
    [2, 2, 2, 2, 2, 2],   # last token 2; earlier pair (2, 2)
]
repeated_bigram_ban(ids, V=6)
# [[False, False,  True, False,  True, False],    # bans 2 and 4
#  [ True, False, False,  True, False, False],    # bans 0 and 3
#  [False, False, False, False, False, False],    # bans nothing
#  [False, False,  True, False, False, False]]    # bans 2
```

## Constraints

- `common_prefix_len` and `repeated_bigram_ban` are checked for exact equality against the reference. `segment_means` is checked with relative tolerance `1e-4` and absolute tolerance `1e-4`.
- Outside the speed tests, $1 \le N \le 64$, and `D`, `S`, `B`, `T`, and `V` are each between 1 and 16. Any dimension can be 1, and any two dimensions can be equal.
- Entries of `x` lie in $[-2, 2]$ for random tests and are whole numbers up to 90 otherwise. Token IDs for `common_prefix_len` lie in $[0, 1000)$.
- **Speed tests.** Each function is timed against a reference in the same framework and process (interleaved runs, median of 3 after one warm-up call). Your implementation must take at most **20 times** as long as the reference, and its output at these sizes is also verified.
  - `segment_means` with `N = 4096` (`1024` in JAX), `D = 16`, `S = 8` in PyTorch and JAX. This case is skipped in MLX because MLX's adding scatter is slow enough that a row loop runs within 3x to 10x of the reference.
  - `common_prefix_len` with `B = 2048`, `T = 16`.
  - `repeated_bigram_ban` with `B = 2048` (`512` in JAX), `T = 16`, `V = 16`.
- A Python loop over rows took more than 75x the reference in every case where the speed test runs. Across roughly forty vectorized formulations tested in all three frameworks, most ran under 5x and the slowest (paired indexing with `torch.unique` in PyTorch) ran at 12x.

## Why this matters

All three patterns appear inside production LLM inference loops. `common_prefix_len` computes the accepted token count in speculative decoding, where draft tokens are compared against target model outputs across a batch. The adding scatter in `segment_means` powers token frequency penalties and Mixture-of-Experts dispatch, where outputs from multiple experts are accumulated back into their original token slots. `repeated_bigram_ban` implements no-repeat n-gram blocking for bigrams, a standard logit processor alongside repetition penalties. Because these operations run for every sequence at every decoding step, a Python loop over the batch adds fixed per-row dispatch overhead on every generated token.
