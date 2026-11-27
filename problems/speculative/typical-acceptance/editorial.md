## The idea

Speculative decoding uses three main verifiers, each with a different guarantee. Greedy verification accepts a draft token only when it matches the target's argmax, reproducing greedy decoding. Speculative sampling accepts with probability $\min(1, p/q)$ for target probability $p$ and draft probability $q$, reproducing exact sampling from the target distribution. Typical acceptance guarantees neither. Instead, it asks whether the target model considers the draft token plausible and checks its target probability against an adaptive threshold:

$$
p(x) > \min\big(\epsilon,\ \delta\, e^{-H(p)}\big).
$$

The two terms inside the minimum handle two regimes.

- **Confident target.** When entropy $H$ is small, $\delta e^{-H}$ is close to $\delta$, so the hard threshold $\epsilon$ usually governs. Any token with probability above $\epsilon$ is a plausible continuation.
- **Uncertain target.** The quantity $e^{-H}$ is the inverse of the distribution's *perplexity*, which measures the effective number of choices. For a uniform distribution over $n$ tokens, $e^{-H} = 1/n$. When the target spreads probability mass across many tokens, no single token may exceed $\epsilon$, yet many are reasonable. Scaling the threshold by $1/\text{perplexity}$ accepts tokens whose probability is at least a fraction $\delta$ of a typical choice.

Using `min` ensures the entropy term can only relax the bar below $\epsilon$ and never raise it above $\epsilon$. This expression originates from truncation sampling, where it filters out unlikely tokens. Here it selects which draft tokens are plausible enough to keep.

After testing each draft token, verification applies the same prefix rule as greedy verification. Rows after the first failure were conditioned on a rejected token and are no longer valid. The row at the first failure is still valid, so the target contributes its greedy token there. In Medusa's formulation, that token is unconditionally accepted at the start of the next step, guaranteeing at least one token of progress per round.

## Reference solution (PyTorch)

```python
def typical_accept(p, draft_tokens, epsilon, delta):
    k = draft_tokens.shape[0]
    entropy = -(p * torch.log(torch.where(p > 0, p, torch.ones_like(p)))).sum(dim=-1)
    threshold = torch.clamp(delta * torch.exp(-entropy), max=epsilon)
    prob = p[torch.arange(k), draft_tokens]
    accepted = (prob > threshold[:k]).to(torch.int64)
    n_accepted = int(torch.cumprod(accepted, dim=0).sum())
    return n_accepted, int(torch.argmax(p[n_accepted]))
```

The JAX and MLX implementations follow the same structure, using `jnp.minimum` or `mx.minimum` for the threshold clamp.

**Handling zero probabilities.** Although $\lim_{p \to 0} p \ln p = 0$, floating-point `0 * log(0)` evaluates to `0 * -inf`, which is `NaN`. Replacing `p` with `1` inside the logarithm wherever `p == 0` turns those terms into `0 * 0 = 0`. Adding a small constant inside `log` also avoids `NaN`, but introduces entropy bias and requires choosing a constant.

**Computing the accepted prefix.** Taking `cumprod` over the 0/1 pass flags produces `1` up to the first failure and `0` thereafter. Summing that cumulative product gives `n_accepted`, which is also the row index that supplies `next_token`.

## Common mistakes

- **Computing entropy in bits.** Using `log2` produces $H / \ln 2$, roughly $1.44\times$ too large. That makes $e^{-H}$ too small and causes uncertain rows to accept tokens they should reject. The factor $e^{-H}$ equals inverse perplexity only when $H$ is in nats.
- **Using `max` instead of `min`.** $\max(\epsilon, \delta e^{-H})$ never relaxes the threshold below $\epsilon$. When $\delta > \epsilon$, it even raises the bar on confident rows, doing the opposite of what is intended.
- **Counting passes after a failure.** Calling `accepted.sum()` directly counts passing tokens beyond the first rejection, even though those rows were computed under an invalid context.
- **Using `>=` instead of `>`.** The pass condition is a strict inequality. Equality matters whenever a probability lands exactly on $\epsilon$, which happens routinely with quantized or rounded probabilities.
- **Evaluating `0 * log(0)`.** A single masked zero probability turns a row's entropy and threshold into `NaN`. Because any comparison with `NaN` is `False`, the row silently rejects every draft token.
- **Indexing the wrong row for the threshold.** Row `i + 1` is the distribution *after* `draft_tokens[i]`. Each draft token must be tested against row `i`, the distribution it was proposed for.
- **Testing `argmax(p)` instead of the probability threshold.** Comparing `draft_tokens` to `argmax(p)` implements greedy verification, never accepting plausible non-greedy alternatives.
- **Reducing entropy over the entire matrix.** Omitting `dim=-1` collapses all rows into one scalar entropy instead of computing one threshold per position.
- **Returning a draft token as `next_token`.** The token following the accepted prefix always comes from the target model (`argmax` of row `n_accepted`), never from `draft_tokens`.

## Complexity

Computing row-wise entropies takes $O(K V)$ time and memory. All remaining operations take $O(K)$ time. Both are negligible compared to the target model's forward pass over the `K` draft tokens.

## In the wild

**Medusa.** The paper *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (arXiv 2401.10774, Section 2.3.1) defines typical acceptance with this exact condition, strict inequality, longest accepted prefix, and unconditionally accepted greedy token. In the reference implementation (`evaluate_posterior` in `medusa/model/utils.py`), the threshold is computed as `minimum(posterior_threshold, exp(-entropy) * posterior_alpha)` and checked via `candidates_prob > threshold`, with `cumprod(...).sum()` giving the accepted length. `medusa_generate` defaults to $\epsilon = 0.09$ and $\delta = 0.3$, noting $\delta \approx \sqrt{\epsilon}$ as a rule of thumb.

Three details differ between full Medusa and this problem. Medusa verifies multiple candidate paths from a token tree at once and selects the longest accepted prefix (breaking ties by likelihood), whereas this problem verifies a single candidate sequence. Medusa also computes `p` by applying softmax to temperature-scaled logits, and avoids `log(0)` by adding `1e-5` inside the logarithm instead of masking zeros.

**What "lossy" means in practice.** At temperature 0, the target distribution is one-hot, only the argmax has non-zero probability, and typical acceptance reduces to greedy verification. At higher temperatures, more tokens clear the threshold and accepted prefixes grow longer, precisely where exact speculative sampling slows down. The resulting output mixes draft tokens the target found typical with greedy target tokens after each rejection. While the Medusa paper reports baseline-comparable quality at its default thresholds, that is an empirical result for specific hyperparameters rather than a mathematical guarantee, so deployments must evaluate output quality alongside speedup.
