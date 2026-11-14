## The idea

A single speculative step turns one sample from the draft distribution $q$ into one
sample from the target distribution $p$: accept $x$ with probability
$\min(1, p[x]/q[x])$, otherwise draw from the normalized residual
$\max(0, p - q)$. The accepted mass is $\min(p, q)$, the residual supplies the
missing $\max(0, p - q)$, and the two add up to $p$.

A block applies that step to $K$ positions that were drafted in a chain, and the
chain is what dictates the rules:

- **Position 0** is an ordinary step. Its output, the accepted draft or the
  correction, is a sample from $p_0$.
- **Position $i$ is meaningful only if drafts $0..i-1$ were accepted.** Both $q_i$
  and $p_i$ are conditioned on the drafted tokens before them. If those tokens are
  exactly what was emitted, then $x_i \sim q_i$ and $p_i$ is the target's true
  conditional, and the step at position $i$ emits a sample from $p_i$. After a
  rejection at $n$, the emitted sequence continues with a different token, so
  $x_{n+1}, \dots$ and their rows describe a continuation that does not exist. They
  must be dropped, whatever their own tests say.
- **Row $K$** is the target's conditional after all $K$ drafts. If every draft was
  accepted it is valid, and no draft token exists for that position. Sampling from
  $p_K$ directly is, trivially, a sample from $p_K$. The residual exists to repair
  a proposal; here there is none.

By induction every emitted token is distributed as the target's conditional given
the tokens emitted before it, so the emitted sequence is a sample from the target
model. That is the theorem of both source papers, and it is what the statistical
cases measure: the token at position $j$, over the calls that got that far, follows
$p_j$.

Independence matters as much as the formulas. Each acceptance test needs its own
uniform, and the resampling draw needs one that is independent of all of them. Given
that position $n$ was rejected, $u_\text{accept}[n]$ is known to lie above
$p_n[x_n]/q_n[x_n]$; it is no longer uniform and cannot be used again.

## Reference solution (PyTorch)

```python
def verify_block(p, q, draft_tokens, u_accept, u_resample):
    k = q.shape[0]
    rows = torch.arange(k)
    passes = u_accept * q[rows, draft_tokens] < p[rows, draft_tokens]
    n_accepted = int(torch.cumprod(passes.to(torch.int64), dim=0).sum())
    if n_accepted < k:
        weights = torch.clamp(p[n_accepted] - q[n_accepted], min=0.0)
    else:
        weights = p[k]
    cdf = torch.cumsum(weights, dim=0)
    next_token = int((cdf <= u_resample * cdf[-1]).sum())
    return n_accepted, next_token
```

All $K$ tests are evaluated at once; the cumulative product turns "passes its own
test" into "passes, and so did everything before". Evaluating tests that end up
ignored is harmless: only the count of the leading run is used. The weights are
never normalized explicitly, because comparing the cumulative sum with
`u_resample * cdf[-1]` is the same test. The JAX and MLX versions are line-by-line
translations.

## Common mistakes

- **Continuing after a rejection.** Counting the drafts that pass (`passes.sum()`),
  or looping without a `break`, keeps tokens that were drafted on top of a rejected
  token. The output then mixes in sequences only the draft model would write.
- **Sampling the bonus token from a residual.** A single code path
  `max(0, p[n] - q[...])` is tempting, with the last draft row standing in for the
  missing `q[K]`. The bonus position has no proposal: subtracting anything removes
  tokens the target wants. (A row of zeros for `q[K]` is the correct way to unify
  the two cases.)
- **Taking the bonus from the wrong row.** `p[K - 1]` is the distribution of the
  last *draft* position, which has already produced its token. The bonus is row `K`,
  the reason the target is given `K + 1` positions to score.
- **Resampling from `p[n]`.** After a rejection the correction comes from the
  residual; the full distribution double-counts the mass that acceptance already
  delivered.
- **Reusing a uniform.** Resampling with `u_accept[n]` biases the correction toward
  tokens late in the vocabulary order, because that uniform is conditioned on having
  caused a rejection.
- **Drawing random numbers inside the function.** The result is then not
  reproducible, and in a real engine not testable; uniforms come from the caller's
  generator.

## Complexity

$O(K)$ for the tests and $O(V)$ for the one sample: $O(K + V)$ time and memory per
block, negligible next to the target's forward pass over $K + 1$ positions.

## In the wild

This is Algorithm 1 of Leviathan, Kalman, and Matias, "Fast Inference from
Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192), whose notation
the problem follows: $\gamma$ guesses $x_i \sim q_i$, the target's
$p_1, \dots, p_{\gamma+1}$ computed in parallel, uniforms $r_1, \dots, r_\gamma$,
$n = \min(\{i - 1 : r_i > p_i(x_i)/q_i(x_i)\} \cup \{\gamma\})$, and one token $t$
sampled from $p_{n+1}$, replaced by $\operatorname{norm}(\max(0, p_{n+1} - q_{n+1}))$
when $n < \gamma$. Algorithm 2 of Chen, Borgeaud, Irving, Lespiau, Sifre, and
Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling"
(2023, arXiv:2302.01318) is the same procedure with the letters exchanged ($q$ is
the target, $p$ the draft, lookahead $K$): accept while $r < \min(1, q/p)$, sample
from $(q - p)_+$ and leave the loop at the first rejection, and if all $K$ tokens
are accepted, sample an extra token from the target. Both papers prove that the
output distribution equals the target's.

Engines implement it batched: `transformers` (`_speculative_sampling` in assisted
generation) and vLLM's rejection sampler compute the acceptance mask for every
sequence and position at once, find the first rejection per row with a cumulative
operation, and gather the residual or bonus row. The distributions `p` and `q` are
those *after* temperature, top-k, and top-p have been applied to each model's
logits, which is why entries can be exactly zero. With temperature 0 both become
one-hot and the rule reduces to comparing argmaxes, the greedy verification of
`greedy-speculative-verify`.
