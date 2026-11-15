Implement the two steps that make speculative sampling work on a **batch**: verify
every row's draft tokens at once, and append what each row accepted. Each row is an
independent sequence, so each stops at its own first rejection, draws its
replacement from its own residual, and grows by its own number of tokens. The batch
goes in rectangular and comes out ragged; the substance is keeping every row exact
while the tensor stays a tensor.

## Specification

```python
def verify_batch(p, q, draft_tokens, u_accept, u_resample): ...


# -> (n_accepted, next_tokens)


def advance_batch(tokens, lengths, draft_tokens, n_accepted, next_tokens, pad_id): ...


# -> (new_tokens, new_lengths, cache_lengths)
```

### `verify_batch`

Row $b$ of the batch is one call of `speculative-sampling-block`. Write $x_{b,i}$
for `draft_tokens[b, i]`, sampled by the draft model from `q[b, i]`; `p[b, i]` is
the target's distribution at the same position, and `p[b, K]` its distribution
after the last draft. Rows share nothing: no result of row $b$ depends on another
row.

**1. Accept a prefix, per row.** Draft $i$ of row $b$ passes its test when

$$
u_\text{accept}[b, i] \cdot q_{b,i}[x_{b,i}] < p_{b,i}[x_{b,i}],
$$

and `n_accepted[b]` is the number of drafts before the first one that does not
pass ($K$ if all pass):

$$
n_b = \min\big(\{\, i : 0 \le i < K,\; u_\text{accept}[b,i] \cdot q_{b,i}[x_{b,i}] \ge p_{b,i}[x_{b,i}] \,\} \cup \{K\}\big).
$$

**2. One more token, per row.** `next_tokens[b]` is drawn by inverse CDF with
`u_resample[b]` from weights $w_b$:

- **rejection at $n_b < K$:** the residual of the rejected position,
  $w_b = \max(0,\; p_{b,n_b} - q_{b,n_b})$, which the sampling normalizes by its
  own total;
- **all $K$ accepted:** the bonus token, $w_b = p_{b,K}$.

With cumulative sums $c_{b,j} = \sum_{v \le j} w_{b,v}$ and the row's total
$W_b = c_{b,V-1}$,

$$
\texttt{next\_tokens}[b] = \min \{\, j : c_{b,j} > u_\text{resample}[b] \cdot W_b \,\}.
$$

### `advance_batch`

`tokens` is a padded matrix: row $b$ holds a sequence of `lengths[b]` tokens in its
first `lengths[b]` columns. **What is stored in the remaining columns is
arbitrary** (padding, or leftovers of an earlier round) and must not reach the
output. The function appends to every row its accepted drafts and its next token:

$$
\text{row}_b' = \texttt{tokens}[b, :\ell_b] \;\Vert\; \texttt{draft\_tokens}[b, :n_b] \;\Vert\; [\texttt{next\_tokens}[b]],
\qquad \ell_b' = \ell_b + n_b + 1 .
$$

It returns

- `new_tokens`: an integer matrix of shape $(B, \max_b \ell_b')$, exactly as wide
  as the longest new row. Row $b$ holds $\text{row}_b'$ in its first $\ell_b'$
  columns and `pad_id` in **every** column after them.
- `new_lengths`: $\ell_b'$ for every row.
- `cache_lengths`: $\ell_b + n_b$ for every row, the number of leading tokens of
  row $b$ whose KV-cache entries are still valid. The target has processed row
  $b$'s old tokens and all $K$ drafts; the entries of the accepted drafts stay,
  those of the rejected ones must be dropped, and the next token has not been fed
  to any model yet. Each row's cache is rolled back to its own length.

Arguments:

- `p`: float32, `(B, K + 1, V)`; every row of every distribution sums to 1; entries
  may be exactly 0.
- `q`: float32, `(B, K, V)`, with $q_{b,i}[x_{b,i}] > 0$.
- `draft_tokens`: integer, `(B, K)`.
- `u_accept`: float32, `(B, K)`, values in $[0, 1)$.
- `u_resample`: float32, `(B,)`, values in $[0, 1)$; `u_resample[b]` is the only
  uniform used for sampling in row $b$.
- `tokens`: integer, `(B, T)`; `lengths`: integer, `(B,)`, with
  $1 \le \ell_b \le T$.
- `n_accepted`, `next_tokens`: integer, `(B,)`, with $0 \le n_b \le K$.
- `pad_id`: a Python int. It may also occur as a real token inside a row.

All five results are integer tensors.

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

Row 0. Draft 0 is token 1 with $p/q = 2$: accepted. Draft 1 is token 1 with
$p/q = 0.25/0.6 \approx 0.417$, and $0.7 \ge 0.417$: rejected, $n_0 = 1$. The
residual of position 1 is $[0.15, 0, 0.05, 0.15]$ with total 0.35 and cumulative
sums $[0.15, 0.15, 0.20, 0.35]$; the threshold $0.5 \times 0.35 = 0.175$ selects
token 2.

Row 1. Token 0 has $p/q = 0.7/0.6 > 1$ and token 2 has $0.3/0.25 > 1$: both
accepted, $n_1 = 2 = K$. The bonus token is sampled from `p[1, 2]` with cumulative
sums $[0.1, 0.3, 0.6, 1.0]$: the first one above 0.45 belongs to token 2.

```python
# row 0 holds 3 tokens (the 77 is a leftover), row 1 holds 4
tokens = [[11, 12, 13, 77], [21, 22, 23, 24]]
advance_batch(tokens, [3, 4], [[1, 1], [0, 2]], [1, 2], [2, 2], pad_id=9)
# new_tokens    = [[11, 12, 13,  1,  2,  9,  9],
#                  [21, 22, 23, 24,  0,  2,  2]]
# new_lengths   = [5, 7]
# cache_lengths = [4, 6]
```

Row 0 gains its one accepted draft and its resampled token and is padded to the
width of row 1, which gains two drafts and its bonus token. The leftover 77 is
gone. The target's cache covered $3 + 2$ and $4 + 2$ tokens after verification;
row 0 is rolled back to 4 (its rejected draft is dropped), row 1 keeps all 6.

## Constraints

- $1 \le B \le 8$, $1 \le K \le 8$, $2 \le V \le 64$, $1 \le T \le 64$.
- **Exact cases.** Single calls of both functions are compared exactly. For
  `verify_batch` their inputs stay away from every decision boundary, so float32
  rounding cannot change the result: $|u_\text{accept}[b,i] - p/q| \ge 10^{-3}$
  for every examined draft with $0 < p/q < 1$, and
  $|u_\text{resample}[b] - c_{b,j} / W_b| \ge 10^{-3}$ for every $j$ with
  $c_{b,j} > 0$.
- **Property.** A row's result must be the same in the batch and alone ($B = 1$).
- **Statistical cases.** The test samples every draft token from its row of `q`,
  draws independent uniforms, and makes up to 400 calls with $B = 3$. Per row it
  compares the distribution of `n_accepted`, of the first emitted token (with
  `p[b, 0]`), and of the bonus token (with `p[b, K]`), using finite-sample
  Chernoff bounds. A correct solution fails with probability below $10^{-9}$, and
  since the seeds are fixed, it either always passes or always fails.
- Both functions are expected to be a fixed number of tensor operations, with no
  Python loop over rows or positions.
- Do not draw random numbers yourself.

## Why this matters

A server does not decode one sequence at a time, and speculative decoding does not
combine with batching for free. After one verification pass, one row has accepted
all five drafts, its neighbor none. The cheap way out is to cut every row back to
the shortest accepted prefix: the tensors stay rectangular, the output stays
exact, and the speedup of the whole batch is set by its unluckiest row, shrinking
toward nothing as the batch grows. Keeping each row's own acceptance means ragged
lengths, per-row cache rollback, and padding that must never be mistaken for
content, which is why batched speculative decoding took engines much longer to
ship than the single-sequence version. Every piece here fails quietly: a residual
normalized by the wrong row's total or a uniform shared across rows changes the
output distribution; a stale token left behind a row's end is read as context
three rounds later.
