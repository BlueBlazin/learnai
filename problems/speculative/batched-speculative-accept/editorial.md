## The idea

Losslessness in speculative sampling guarantees that a single sequence's emitted
tokens follow the exact distribution of the target model. A batch of $B$
sequences requires $B$ simultaneous guarantees, which hold as long as **nothing in
row $b$'s computation reads another row**. With a Python loop over rows, isolation
is automatic. With vectorized tensor operations, every reduction can accidentally
cross the batch axis, and values that were scalars in the single-sequence version
(the stopping position, the residual total, and the resampling uniform) become
length-$B$ vectors.

`verify_batch` extends `speculative-sampling-block` by promoting three
single-sequence scalars to per-row values:

- **Per-row stopping position.** The boolean acceptance matrix `passes` has shape
  `(B, K)`. Taking a cumulative product along the draft axis (`dim=1`) zeroes out
  every position after the first failure, and summing along the same axis yields
  $n_b$ for each row. Reducing across the whole matrix or taking a minimum across
  rows would break per-row acceptance.
- **Per-row, per-position residual.** Row $b$ needs $\max(0, p_{b,n_b} - q_{b,n_b})$,
  which can be fetched in one advanced-indexing gather such as `p[arange(B), n]`.
  Appending a zero slice to `q` at position $K$ unifies the rejection and bonus
  cases into a single expression, because $\max(0, p_{b,K} - 0) = p_{b,K}$.
- **Per-row normalization and uniform.** Each row's inverse-CDF threshold is
  $u_b \cdot W_b$, where $u_b = \texttt{u\_resample}[b]$ and $W_b$ is row $b$'s own
  residual total. Because a row's residual total equals one minus its acceptance
  probability at the rejected position, totals vary widely across rows, so sharing
  a single total produces large errors.

For any fixed row $b$, these operations compute the exact same result as the
single-sequence algorithm on row $b$'s inputs. Because each row's drafts and
uniforms are independent of the other rows, the single-sequence proof of
losslessness applies verbatim to every row. The property test verifies this
isolation directly, and the statistical test verifies each row's marginal
distribution.

`advance_batch` handles the ragged sequence update, which is a frequent source of
bugs in batched speculation. Rows start with different lengths $\ell_b$ and gain
different numbers of tokens, so there is no single column where the new tokens
belong. Instead, we compute the output width $W = \max_b (\ell_b + n_b + 1)$ and
assign every cell `(b, c)` of the `(B, W)` output matrix according to one of four
disjoint rules:

| condition on column $c$ | value |
|---|---|
| $c < \ell_b$ | old token |
| $\ell_b \le c < \ell_b + n_b$ | draft $c - \ell_b$ |
| $c = \ell_b + n_b$ | next token |
| $c > \ell_b + n_b$ | `pad_id` |

Because every cell is written by exactly one rule, no leftover tokens past the old
length and no rejected drafts can survive. `lengths` is the sole source of truth
for sequence boundaries, since `pad_id` may also appear as a legitimate token.

The third return value, `cache_lengths`, specifies how far to roll back each row's
KV cache. After verification, the target's cache holds each row's $\ell_b$ old
tokens plus all $K$ drafts. Entries in $[0, \ell_b + n_b)$ are verified and stay,
while entries from $\ell_b + n_b$ onward belong to rejected drafts and are dropped.
The newly sampled `next_tokens[b]` has not been processed yet and will be the input
to the next forward pass. Carrying a per-row cache length (or attention mask) lets
the KV-cache tensor stay rectangular while each row ignores positions beyond its
own valid length.

## Reference solution (PyTorch)

```python
def verify_batch(p, q, draft_tokens, u_accept, u_resample):
    b, k = draft_tokens.shape
    index = draft_tokens[:, :, None]
    p_x = torch.gather(p[:, :k], 2, index)[:, :, 0]
    q_x = torch.gather(q, 2, index)[:, :, 0]
    passes = u_accept * q_x < p_x
    n_accepted = torch.cumprod(passes.to(torch.int64), dim=1).sum(dim=1)
    q_ext = torch.cat([q, torch.zeros_like(q[:, :1])], dim=1)
    rows = torch.arange(b)
    weights = torch.clamp(p[rows, n_accepted] - q_ext[rows, n_accepted], min=0.0)
    cdf = torch.cumsum(weights, dim=1)
    threshold = u_resample[:, None] * cdf[:, -1:]
    return n_accepted, (cdf <= threshold).sum(dim=1)


def advance_batch(tokens, lengths, draft_tokens, n_accepted, next_tokens, pad_id):
    cache_lengths = lengths + n_accepted
    new_lengths = cache_lengths + 1
    width = int(new_lengths.max())
    cols = torch.arange(width)[None, :]
    start, stop = lengths[:, None], cache_lengths[:, None]
    old = torch.gather(tokens, 1, cols.clamp(max=tokens.shape[1] - 1).expand(len(lengths), -1))
    drafts = torch.gather(draft_tokens, 1, (cols - start).clamp(0, draft_tokens.shape[1] - 1))
    new_tokens = torch.full_like(old, pad_id)
    new_tokens = torch.where(cols == stop, next_tokens[:, None], new_tokens)
    new_tokens = torch.where((cols >= start) & (cols < stop), drafts, new_tokens)
    new_tokens = torch.where(cols < start, old, new_tokens)
    return new_tokens, new_lengths, cache_lengths
```

The JAX and MLX versions are identical except they use `take_along_axis` in place
of `gather`.

## Common mistakes

- **Truncating every row to the batch minimum.** Setting every $n_b$ to
  $\min_b n_b$ keeps outputs exact, which makes it easy to miss in code review, but
  it destroys throughput. With per-position acceptance rate $\alpha$, one row
  accepts all $K$ drafts with probability $\alpha^K$, whereas a batch of $B$ rows
  all accept $K$ drafts with probability $\alpha^{KB}$. At $\alpha = 0.8$ and
  $K = 4$, that drops from 41% for a single sequence to 3% for a batch of four.
- **Counting passes instead of the leading run.** Using `passes.sum(dim=1)` keeps
  drafts that were generated on top of an already-rejected token.
- **Computing the bonus from a residual or the correction from `p`.** Reusing the
  last draft distribution for the missing `q[:, K]` corrupts the bonus token, and
  sampling a rejection directly from `p` instead of $\max(0, p - q)$ breaks
  losslessness.
- **Using one residual total for the whole batch.** Taking `cdf.max()` or
  `cdf[-1, -1]` instead of `cdf[:, -1:]` leaves rows with small residuals
  unnormalized, pushing their samples toward the end of the vocabulary (or past it).
- **Sharing one uniform across the batch.** Using `u_resample[0]` for every row
  preserves each row's marginal distribution, so per-row statistical tests will not
  catch it, but it couples different requests in the batch and fails the exact
  tests.
- **Appending at the batch width $T$.** Concatenating new tokens onto the right
  edge of `tokens` places them after the padding of shorter rows instead of right
  after `lengths[b]`.
- **Leaving rejected drafts or old leftovers past the new end.** Stale tokens may
  look harmless until a later round extends into them or a downstream consumer
  inspects `pad_id`.
- **Miscounting `next_tokens` in the lengths.** `new_lengths` must include the next
  token ($\ell_b + n_b + 1$), while `cache_lengths` must not ($\ell_b + n_b$).

## Complexity

Let $W = \max_b (\ell_b + n_b + 1)$ be the width of `new_tokens`. `verify_batch`
runs in $O(BK + BV)$ time, and `advance_batch` runs in $O(B \cdot W)$ time. Both
use a constant number of tensor operations and are negligible compared to the
target model's forward pass over $B(K + 1)$ positions.

## In the wild

- The per-sequence rule is Algorithm 1 of "Fast Inference from Transformers via
  Speculative Decoding" (ICML 2023, arXiv:2211.17192). See
  `speculative-sampling-block`.
- "BASS: Batched Attention-optimized Speculative Sampling" (arXiv:2404.15778)
  targets this exact bottleneck. Because accepted lengths vary across a batch, the
  paper introduces ragged attention kernels alongside a per-step heuristic for
  choosing the draft length.
- Serving engines make different trade-offs here. Some track per-row lengths and
  attention masks as in this problem, some enable speculation only at batch size 1,
  and others truncate to the batch minimum and accept the lower speedup. When
  auditing batched speculation code, check the reduction axis for three quantities:
  first rejection, residual total, and resampling uniform.
