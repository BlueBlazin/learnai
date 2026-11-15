## The idea

Speculative decoding has three verifiers, and they differ in what they promise.
Greedy verification accepts a draft token only if it is the target's argmax, and
reproduces the target's greedy decode. Speculative sampling accepts with probability
$\min(1, p/q)$ and reproduces the target's sampling distribution. Typical acceptance
promises neither. It asks a weaker question, "would the target plausibly have said
this?", and answers it with the target's own probability of the token:

$$
p(x) > \min\big(\epsilon,\ \delta\, e^{-H(p)}\big).
$$

The two terms cover two regimes.

- **The target is confident.** The entropy $H$ is small, $\delta e^{-H}$ is close to
  $\delta$, and the hard threshold $\epsilon$ decides: tokens with a probability
  above $\epsilon$ are meaningful continuations.
- **The target is uncertain.** $e^{-H}$ is the reciprocal of the *perplexity* of the
  distribution, the effective number of choices: a uniform distribution over $n$
  tokens has $e^{-H} = 1/n$. When the target spreads its mass over many tokens, none
  of them reaches $\epsilon$, yet each is a reasonable continuation. Scaling the bar
  with $1/\text{perplexity}$ accepts a token that has at least a fraction $\delta$
  of the probability of a "typical" choice.

The `min` makes the entropy term a relaxation only: it can lower the bar below
$\epsilon$ and never raises it above. The criterion comes from the truncation
sampling literature, where the same expression decides which tokens are too unlikely
to sample; here it decides which draft tokens are likely enough to keep.

Everything else is the prefix logic of greedy verification. Rows after the first
failure were conditioned on a token that is no longer part of the sequence, so they
say nothing about it. And the row at the first failure is still valid, so the target
contributes its own greedy token there. In Medusa's formulation that token is the
first element of the next round's candidate and is "unconditionally accepted", which
guarantees progress: a round yields at least one token even if every draft fails.

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

The JAX and MLX versions are the same lines with `jnp.minimum` / `mx.minimum`.

**The zero probabilities.** $\lim_{p \to 0} p \ln p = 0$, but in floating point
`0 * log(0)` is `0 * -inf`, which is NaN. Replacing the argument of the logarithm by
1 where `p` is 0 makes the term `0 * 0`. Adding a small constant inside the
logarithm also avoids the NaN, at the price of a biased entropy; with a mask no
constant has to be chosen.

**The running product.** `cumprod` of the pass flags is 1 up to the first failure
and 0 afterwards, so its sum is the length of the accepted prefix, and it doubles as
the index of the row that supplies the next token.

## Common mistakes

- **Entropy in bits.** `log2` gives $H / \ln 2$, about 1.44 times too large, so
  $e^{-H}$ is too small and uncertain rows accept tokens they should reject. The
  formula's $e^{-H}$ is the inverse perplexity only with the natural logarithm.
- **`max` for `min`.** $\max(\epsilon, \delta e^{-H})$ never drops below $\epsilon$,
  so the bar is not relaxed when the target is uncertain, and with $\delta > \epsilon$
  it is raised when the target is confident: the opposite of the intent.
- **Counting after a failure.** `accepted.sum()` counts every token that passes. The
  tokens after the first failure were scored in a context that no longer exists.
- **`>=`.** The criterion is a strict inequality. It matters exactly when a
  probability equals $\epsilon$, which is not exotic with quantized or rounded
  probabilities.
- **`0 * log(0)`.** One masked token turns the row's entropy, its threshold, and
  therefore every comparison into NaN, and a comparison with NaN is false: the row
  silently rejects everything.
- **The threshold of the wrong row.** Row `i + 1` is the distribution *after* draft
  token `i`. The token must be tested against the distribution it was proposed for,
  row `i`.
- **Testing the argmax.** Comparing draft tokens with `argmax(p)` is greedy
  verification. It passes every case in which the draft equals the target's choice,
  and never accepts the plausible alternatives this scheme exists for.
- **One entropy for the matrix.** A reduction without an axis gives a single number
  for all rows; each position has its own distribution and its own bar.
- **Returning the draft token.** The token after the accepted prefix comes from the
  target (`argmax` of row `n`), never from the draft.

## Complexity

$O(K V)$ time and memory for the entropies; everything else is $O(K)$. The cost that
matters, the target's forward pass over the `K` draft tokens, has already been paid.

## In the wild

**Medusa** ("Medusa: Simple LLM Inference Acceleration Framework with Multiple
Decoding Heads", arXiv 2401.10774, section 2.3.1) introduces the scheme with exactly
this condition, the strict inequality, the unconditionally accepted greedy first
token, and the longest accepted prefix. Its reference code
(`evaluate_posterior` in `medusa/model/utils.py`) computes
`threshold = minimum(posterior_threshold, exp(-entropy) * posterior_alpha)` and
`candidates_prob > threshold`, takes `cumprod(...).sum()` as the accepted length, and
uses defaults $\epsilon = 0.09$ and $\delta = 0.3$ in `medusa_generate`, with the
comment that $\delta$ is recommended to be $\sqrt{\epsilon}$. Three things differ
from this problem. Medusa verifies **several** candidates at once (the paths of a
token tree) and keeps the one with the longest accepted prefix, breaking ties by
likelihood; this problem is one candidate. Its probabilities are the softmax of the
logits divided by the sampling temperature; here `p` is given. And its entropy adds
`1e-5` inside the logarithm instead of masking zeros.

**What "lossy" means in practice.** With a temperature of 0 the target's
distribution is one-hot, only the argmax has a non-zero probability, and the scheme
reduces to greedy verification. As the temperature rises, more tokens clear the bar
and accepted runs get longer, which is the regime in which exact speculative
sampling slows down. The output is then a mixture: drafted tokens that the target
found typical, and the target's greedy token after each failure. The Medusa paper
reports generation quality similar to the baseline at its chosen thresholds; that is
an empirical statement about two hyperparameters, not a guarantee, so a system that
ships this needs an evaluation of output quality next to its speedup number.
