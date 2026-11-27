Implement `verify_batch` and `advance_batch` to run speculative sampling across a
batch of sequences. Each row in the batch is an independent sequence, so every row
stops at its own first rejection, samples its own next token, and grows to its own
length.

![This problem is Roll back caches, where each row of a batch keeps its own number of tokens.](assets/map.svg)

## Specification

```python
def verify_batch(p, q, draft_tokens, u_accept, u_resample): ...
# -> (n_accepted, next_tokens)


def advance_batch(tokens, lengths, draft_tokens, n_accepted, next_tokens, pad_id): ...
# -> (new_tokens, new_lengths, cache_lengths)
```

### `verify_batch`

A batch contains $B$ independent sequences, each with $K$ drafted tokens over a
vocabulary of size $V$. For row $b \in \{0, \dots, B - 1\}$ and draft position
$i \in \{0, \dots, K - 1\}$, write $x_{b,i} = \texttt{draft\_tokens}[b, i]$ for the
token sampled by the draft model from distribution $q_{b,i} = \texttt{q}[b, i]$.
The target model's distribution at that same position is $p_{b,i} = \texttt{p}[b, i]$,
and its distribution after all $K$ draft tokens is $p_{b,K} = \texttt{p}[b, K]$.
Every distribution in `p` and `q` sums to 1 (entries may be 0), and
$q_{b,i}[x_{b,i}] > 0$. Rows are completely independent, so nothing computed for
row $b$ may depend on any other row.

Each row runs the same two-step acceptance and resampling rule as
`speculative-sampling-block`.

**1. Accept a leading prefix per row.** Using the uniform draw `u_accept[b, i]` in
$[0, 1)$, draft $i$ of row $b$ passes its rejection test when

$$
u_\text{accept}[b, i] \cdot q_{b,i}[x_{b,i}] < p_{b,i}[x_{b,i}].
$$

The accepted count `n_accepted[b]` (written $n_b$) is the index of the first
failing draft, or $K$ if all $K$ drafts pass:

$$
n_b = \min\big(\{\, i : 0 \le i < K,\; u_\text{accept}[b,i] \cdot q_{b,i}[x_{b,i}] \ge p_{b,i}[x_{b,i}] \,\} \cup \{K\}\big).
$$

**2. Sample one next token per row.** After accepting $n_b$ drafts, row $b$ draws
one additional token `next_tokens[b]` by inverse-CDF sampling from unnormalized
weights $w_b$ of length $V$:

- **Rejection ($n_b < K$):** use the clamped residual at the rejected position,
  $w_b = \max(0,\; p_{b,n_b} - q_{b,n_b})$.
- **All $K$ accepted ($n_b = K$):** draw a bonus token from the final target
  distribution, $w_b = p_{b,K}$.

Form the cumulative sums $c_{b,j} = \sum_{v=0}^{j} w_{b,v}$ for $0 \le j < V$, and
let $W_b = c_{b,V-1}$ be row $b$'s total weight. Using `u_resample[b]` in $[0, 1)$
(the only uniform used for sampling in row $b$), select the first token whose
cumulative sum strictly exceeds the scaled threshold:

$$
\texttt{next\_tokens}[b] = \min \{\, j : c_{b,j} > u_\text{resample}[b] \cdot W_b \,\}.
$$

### `advance_batch`

`tokens` is a padded matrix of shape `(B, T)` holding the current sequences. Row
$b$ stores its valid sequence of length $\ell_b = \texttt{lengths}[b]$
($1 \le \ell_b \le T$) in columns $0$ through $\ell_b - 1$. Values in the remaining
columns are arbitrary padding or leftovers from earlier rounds and must never reach
the output.

For each row $b$, append its $n_b = \texttt{n\_accepted}[b]$ accepted drafts
($0 \le n_b \le K$) followed by `next_tokens[b]`:

$$
\text{row}_b' = \texttt{tokens}[b, :\ell_b] \;\Vert\; \texttt{draft\_tokens}[b, :n_b] \;\Vert\; [\texttt{next\_tokens}[b]],
\qquad \ell_b' = \ell_b + n_b + 1 .
$$

Return three integer tensors:

- `new_tokens`: an integer matrix of shape $(B, \max_b \ell_b')$, sized to the
  longest updated row. Row $b$ holds $\text{row}_b'$ in its first $\ell_b'$ columns
  and the Python integer `pad_id` in every column after them. Note that `pad_id`
  may also occur as a real token inside a sequence, so always rely on `lengths`
  rather than `pad_id` to find a row's end.
- `new_lengths`: $\ell_b'$ for every row.
- `cache_lengths`: $\ell_b + n_b$ for every row, giving the number of leading
  tokens whose KV-cache entries remain valid. During verification, the target model
  processed row $b$'s $\ell_b$ existing tokens plus all $K$ drafts. The entries for
  the $n_b$ accepted drafts stay, the entries for rejected drafts must be dropped,
  and `next_tokens[b]` has not been passed through any model yet. Each row's cache
  is rolled back to its own `cache_lengths[b]`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(B, K + 1, V)` | float32 |
| `q` | `(B, K, V)` | float32 |
| `draft_tokens` | `(B, K)` | integer |
| `u_accept` | `(B, K)` | float32 |
| `u_resample` | `(B,)` | float32 |
| `n_accepted`, `next_tokens` | `(B,)` | integer |
| `tokens` | `(B, T)` | integer |
| `lengths`, `new_lengths`, `cache_lengths` | `(B,)` | integer |
| `new_tokens` | `(B, max(new_lengths))` | integer |

## Example

```python
p = [
    # row 0: at draft 0, at draft 1, and after the block
    [[0.10, 0.40, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], [0.50, 0.10, 0.10, 0.30]],
    # row 1
    [[0.70, 0.10, 0.10, 0.10], [0.20, 0.20, 0.30, 0.30], [0.10, 0.20, 0.30, 0.40]],
]
q = [
    [[0.50, 0.20, 0.20, 0.10], [0.10, 0.60, 0.20, 0.10]],
    [[0.60, 0.20, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
]

verify_batch(p, q, [[1, 1], [0, 2]], [[0.9, 0.7], [0.8, 0.6]], [0.5, 0.45])
# n_accepted = [1, 2], next_tokens = [2, 2]
```

**Row 0.** Draft 0 is token 1 with $p/q = 0.40 / 0.20 = 2.0$, so it passes. Draft 1
is token 1 with $p/q = 0.25 / 0.60 \approx 0.417$, and $0.7 \ge 0.417$, so it is
rejected and $n_0 = 1$. The residual at position 1 is $[0.15, 0, 0.05, 0.15]$,
which has total weight $0.35$ and cumulative sums $[0.15, 0.15, 0.20, 0.35]$. The
scaled threshold is $0.5 \times 0.35 = 0.175$, which selects token 2.

**Row 1.** Token 0 has $p/q = 0.70 / 0.60 > 1$ and token 2 has
$p/q = 0.30 / 0.25 > 1$, so both pass and $n_1 = 2 = K$. The bonus token is sampled
from `p[1, 2]`, whose cumulative sums are $[0.1, 0.3, 0.6, 1.0]$. The first entry
strictly above $0.45$ is at index 2.

```python
# row 0 holds 3 tokens (the 77 is a leftover), row 1 holds 4
tokens = [[11, 12, 13, 77], [21, 22, 23, 24]]
advance_batch(tokens, [3, 4], [[1, 1], [0, 2]], [1, 2], [2, 2], pad_id=9)
# new_tokens    = [[11, 12, 13,  1,  2,  9,  9],
#                  [21, 22, 23, 24,  0,  2,  2]]
# new_lengths   = [5, 7]
# cache_lengths = [4, 6]
```

Row 0 appends its 1 accepted draft and its resampled token, then pads to length 7
to match row 1, which appends 2 accepted drafts and its bonus token. The leftover
token `77` is discarded. After verification, the target's KV cache held $3 + 2 = 5$
and $4 + 2 = 6$ tokens. Row 0 rolls back to 4 (dropping its rejected draft), while
row 1 keeps all 6.

## Constraints

- $1 \le B \le 8$, $1 \le K \le 8$, $2 \le V \le 64$, $1 \le T \le 64$.
- **Exact cases.** Single calls to both functions are compared exactly. For
  `verify_batch`, test inputs stay away from decision boundaries so float32
  rounding cannot flip a result: $|u_\text{accept}[b,i] - p/q| \ge 10^{-3}$ for
  every examined draft with $0 < p/q < 1$, and
  $|u_\text{resample}[b] - c_{b,j} / W_b| \ge 10^{-3}$ for every $j$ with
  $c_{b,j} > 0$.
- **Batch-independence property.** Each row's result in a batch must match its
  result when run alone ($B = 1$).
- **Statistical cases.** The test samples every draft token from its row of `q`,
  draws independent uniforms, and makes up to 400 calls with $B = 3$. For each row,
  it checks the distributions of `n_accepted`, the first emitted token (against
  `p[b, 0]`), and the bonus token (against `p[b, K]`) using finite-sample Chernoff
  bounds. A correct solution fails with probability below $10^{-9}$, and because
  the seeds are fixed, it either always passes or always fails.
- Both functions must use a fixed number of tensor operations, with no Python loop
  over rows or positions.
- Do not draw random numbers yourself.

## Why this matters

Production inference servers decode batches of requests, and speculative decoding
does not combine with batching for free. After one verification pass, one row might
accept all five drafts while its neighbor accepts none. A simple workaround is to
truncate every row to the shortest accepted prefix in the batch. That keeps tensors
rectangular and preserves exactness, but it ties the entire batch's speedup to its
unluckiest row, eroding the gain as batch size grows.

Keeping each row's own accepted count requires ragged lengths, per-row KV-cache
rollback, and strict padding discipline. Subtle bugs here fail silently. Normalizing
a residual by the wrong row's total or sharing a uniform across rows distorts the
output distribution without raising an error, and leaving a stale token past a
row's end can leak into the context several rounds later.
