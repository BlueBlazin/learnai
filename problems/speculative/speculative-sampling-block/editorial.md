## The idea

A single speculative sampling step turns one sample $x$ from a draft distribution $q$ into one sample from a target distribution $p$. It accepts $x$ with probability $\min(1, p[x]/q[x])$ and otherwise draws from the normalized residual $\max(0, p - q)$. Acceptance contributes mass $\min(p, q)$, resampling supplies the missing $\max(0, p - q)$, and their sum equals $p$.

A block applies this step to $K$ tokens drafted in a chain, and that chain determines how the positions interact:

- **Position 0** is a standard single step. Its output, whether accepted or resampled, follows $p_0$.
- **Position $i$ is valid only if drafts $0, \dots, i-1$ were all accepted.** Both $q_i$ and $p_i$ are conditioned on the preceding draft tokens. When those tokens are actually emitted, $x_i \sim q_i$ and $p_i$ is the target's true conditional, so position $i$ emits a sample from $p_i$. Once draft $n$ is rejected, the sequence continues with a different token. Later drafts $x_{n+1}, \dots, x_{K-1}$ and their distributions belong to a continuation that never happened, so they must be discarded even if their individual tests pass.
- **Row $K$** is the target's conditional after all $K$ drafts. When every draft is accepted, this row is valid and has no draft token to correct. Sampling directly from $p_K$ produces a valid bonus token from $p_K$.

By induction, every emitted token follows the target's conditional distribution given the tokens emitted before it, so the whole sequence matches the target model. That is what the statistical tests check by verifying that the token at position $j$, across calls that reach position $j$, follows $p_j$.

Independence of the random draws is just as important as the formulas. Each acceptance test requires its own uniform, and the resampling draw must be independent of all of them. Once position $n$ is rejected, $u_\text{accept}[n]$ is known to be at least $p_n[x_n]/q_n[x_n]$, so it is no longer uniform on $[0, 1)$ and cannot be reused.

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

Evaluating all $K$ acceptance tests in parallel is safe because the cumulative product zeroes out every entry after the first failure. Summing that cumulative product gives the length of the leading run of accepted drafts. We do not need to normalize `weights` explicitly, because comparing `cdf` against `u_resample * cdf[-1]` is equivalent. The JAX and MLX implementations are direct line-by-line translations.

## Common mistakes

- **Continuing after a rejection.** Counting all passing drafts (`passes.sum()`) or looping without breaking keeps tokens drafted after a rejected token. The output then mixes in continuations from the draft model.
- **Sampling the bonus token from a residual.** Using one code path `max(0, p[n] - q[...])` with the last draft row in place of a missing `q[K]` subtracts mass where no proposal exists. If you want to unify both cases, append a row of zeros for `q[K]`.
- **Taking the bonus from the wrong row.** Row `p[K - 1]` belongs to the last draft position, which already produced its token. The bonus token comes from row `p[K]`, which is why the target scores $K + 1$ positions.
- **Resampling from `p[n]` after a rejection.** The correction must come from the residual $\max(0, p_n - q_n)$. Using the full distribution `p[n]` double-counts probability mass that acceptance already covered.
- **Reusing a uniform.** Passing `u_accept[n]` to the inverse CDF biases the resampled token toward later indices in the vocabulary, because that uniform is conditioned on causing a rejection.
- **Drawing random numbers inside the function.** Internal RNG calls break reproducibility and testability. All uniforms must come from the caller.

## Complexity

Computing the $K$ acceptance tests takes $O(K)$ time, and inverse-CDF sampling over the vocabulary takes $O(V)$ time. Total work and memory are $O(K + V)$ per block, which is negligible compared to the target model's forward pass over $K + 1$ positions.

## In the wild

This procedure is Algorithm 1 of Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192). That paper drafts $\gamma$ tokens $x_i \sim q_i$, computes the target distributions $p_1, \dots, p_{\gamma+1}$ in parallel, draws uniforms $r_1, \dots, r_\gamma$, sets $n = \min(\{i - 1 : r_i > p_i(x_i)/q_i(x_i)\} \cup \{\gamma\})$, and samples one token from $p_{n+1}$ when $n = \gamma$ or from $\operatorname{norm}(\max(0, p_{n+1} - q_{n+1}))$ when $n < \gamma$. Algorithm 2 of Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023, arXiv:2302.01318), is the same algorithm with the letters swapped ($q$ for target, $p$ for draft, and lookahead $K$). Both papers prove that the output distribution matches the target model exactly.

Production engines run this logic in batches. Hugging Face `transformers` (`_speculative_sampling` in assisted generation) and vLLM's rejection sampler compute the acceptance mask across all sequences and positions at once, locate the first rejection per row with a cumulative operation, and gather either the residual or the bonus row. The inputs `p` and `q` are distributions *after* temperature scaling, top-k, and top-p filtering, which is why entries can be exactly zero. At temperature 0, both distributions become one-hot and the rule reduces to the argmax comparison in `greedy-speculative-verify`.
