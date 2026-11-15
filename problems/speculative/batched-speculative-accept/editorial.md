## The idea

Losslessness is a statement about one sequence: its emitted tokens are distributed
as the target model would have sampled them. A batch is $B$ such statements, and
the only way to keep all of them is to make sure that **nothing in row $b$'s
computation reads another row**. Written as a loop over rows this is trivial.
Written as tensor operations, every reduction is a chance to reduce over the wrong
axis, and every scalar of the single-sequence code (the stopping position, the
residual's total, the uniform) has silently become a vector.

`verify_batch` is `speculative-sampling-block` with three such promotions:

- **The stopping position is per row.** `passes` is `(B, K)`; the running product
  along the draft axis keeps a row's leading run of `True`, and its sum is
  $n_b$. A reduction over the whole matrix, or a `min` over rows, is the classic
  shortcut that keeps the batch rectangular.
- **The residual is per row and per position.** Row $b$ needs
  $\max(0, p_{b,n_b} - q_{b,n_b})$: a gather with two index vectors,
  `p[arange(B), n]`. Padding `q` with a zero row at position $K$ turns the bonus
  case into the same expression, since $\max(0, p_{b,K} - 0) = p_{b,K}$.
- **The normalization and the uniform are per row.** The threshold is
  $u_b \cdot W_b$ with $W_b$ the row's own total. Residual totals differ wildly
  between rows (it is one minus the acceptance probability at that position), so
  a shared total is not a small error.

Why every row stays exact: for a fixed row, the function computes exactly what the
single-sequence function computes from that row's inputs, and those inputs (its
drafts, its uniforms) are independent of the other rows'. The proof for one
sequence therefore applies to each row verbatim. This is what the property case
checks directly, and the statistical case per row.

`advance_batch` is bookkeeping, and bookkeeping is where batched speculation
usually breaks. Rows had different lengths before the round and gain different
numbers of tokens, so there is no single column at which "the new tokens" go. The
reference decides, for every cell `(b, c)` of the output, which source owns it:

| condition on column $c$ | value |
|---|---|
| $c < \ell_b$ | old token |
| $\ell_b \le c < \ell_b + n_b$ | draft $c - \ell_b$ |
| $c = \ell_b + n_b$ | next token |
| $c > \ell_b + n_b$ | `pad_id` |

Every cell is written from exactly one rule, so nothing stale can survive: not the
leftovers past a row's old length, and not the rejected drafts. The lengths are the
source of truth; `pad_id` may be a real token, so lengths can never be recovered
from the matrix.

The third result, `cache_lengths`, is the rollback instruction. After the
verification pass the target's cache holds, for every row, its old tokens and all
$K$ drafts. Entries $[0, \ell_b + n_b)$ are verified tokens and stay; entries from
$\ell_b + n_b$ on belong to rejected drafts. The next token is not in any cache:
it is the input of the next pass. With a per-row length (or attention mask) the
cache tensor can stay rectangular while each row ignores what lies beyond its own
length; that mask is the price of raggedness.

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

The JAX and MLX versions are the same with `take_along_axis` for `gather`.

## Common mistakes

- **Truncating every row to the batch minimum.** Set every $n_b$ to
  $\min_b n_b$ and let rows that accepted more re-emit their next draft. The
  output is still exact, which is why this survives review; what is lost is
  speed. With per-position acceptance rate $\alpha$, one row accepts all $K$ drafts
  with probability $\alpha^K$ and a batch of $B$ rows with probability
  $\alpha^{KB}$: at $\alpha = 0.8$, $K = 4$, that is 41 percent for one sequence
  and 3 percent for a batch of four.
- **Counting passes instead of the leading run.** `passes.sum(dim=1)` keeps
  drafts that were proposed on top of a rejected token.
- **The bonus from a residual**, with the last draft row standing in for the
  missing `q[:, K]`, or **the correction from `p`** instead of the residual: the
  two single-sequence mistakes, now per row.
- **One total for the batch.** `cdf.max()` or `cdf[-1, -1]` instead of
  `cdf[:, -1:]`: rows with small residuals are not normalized and their sample
  piles up on the last tokens (or runs off the end of the vocabulary).
- **One uniform for the batch.** `u_resample[0]` for every row gives each row the
  right marginal distribution and couples the rows: two requests in one batch
  now share randomness. No per-row statistic sees it; the exact cases do.
- **Appending at the batch width.** Concatenating the new tokens after the old
  matrix puts them at column $T$ for every row, behind the padding of the shorter
  rows.
- **Leaving rejected drafts or old leftovers past the end.** Harmless until a
  later round extends the row over them, or a consumer trusts `pad_id`.
- **Off by the next token.** `new_lengths` must count it; `cache_lengths` must
  not.

## Complexity

`verify_batch` is $O(BK + BV)$ after the gathers, `advance_batch` is
$O(B \cdot W)$; both are a constant number of tensor operations and negligible
next to the target's forward pass over $B (K + 1)$ positions.

## In the wild

- The per-sequence rule is Algorithm 1 of "Fast Inference from Transformers via
  Speculative Decoding" (ICML 2023, arXiv:2211.17192); see
  `speculative-sampling-block`.
- "BASS: Batched Attention-optimized Speculative Sampling" (arXiv:2404.15778)
  is about exactly this situation: the number of accepted draft tokens differs
  across the batch, the tensors become ragged, and the paper handles them with
  attention kernels written for ragged inputs, together with a per-step heuristic
  for the draft length. It is a good account of why batching and speculation
  interact badly.
- Engines differ in where they pay. Some keep every row's own acceptance and carry
  per-row lengths and masks, as here; some run speculation only at batch size one;
  some truncate to the minimum and accept the loss. When reading such code, look
  for the three reductions above (first rejection, residual total, uniform) and
  check the axis of each.
