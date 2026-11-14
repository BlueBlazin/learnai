## The idea

Model each draft token as accepted independently with probability $\alpha$. Because
verification keeps only the prefix before the first rejection, the number of
accepted drafts $n$ is a geometric variable capped at $k$:
$P(n \ge i) = \alpha^i$ for $i \le k$. The round emits $n + 1$ tokens, since the
target's single pass always yields one valid token of its own: the correction at
the first rejection, or the bonus token after $k$ acceptances. By the tail-sum
formula,

$$
E(\alpha, k) = \mathbb{E}[n + 1] = 1 + \sum_{i=1}^{k} P(n \ge i)
= \sum_{i=0}^{k} \alpha^i = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

For the speedup, measure time in target calls. Plain decoding produces one token
per unit. A round costs $k$ draft calls at $c$ units each plus one target call, and
produces $E$ tokens, so

$$
S(\alpha, k, c) = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(k c + 1)} .
$$

This assumes what makes speculative decoding work at all: scoring $k + 1$ positions
in one target pass takes about as long as scoring one, because the pass is batched
over positions on hardware with spare parallel capacity.

The numerator saturates at $1 / (1 - \alpha)$ while the denominator grows linearly,
so $S$ rises, peaks, and falls: there is a best draft length, and it is found by
trying each integer. Setting $k = 1$ shows when speculation can win at all:
$S = (1 + \alpha) / (1 + c) > 1$ exactly when $\alpha > c$. If $\alpha \le c$, every
added term $\alpha^i \le c$ adds less to the numerator than $c$ adds to the
denominator, so no $k$ helps and `best_k` is 0.

## Reference solution (Python)

```python
def expected_tokens(alpha, k):
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha, k, c):
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha, c, k_max):
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
```

`max` returns the first maximal element, which is the smallest `k`, and `k = 0`
gives $E = 1$ and $S = 1$ without a special case.

## Common mistakes

- **Forgetting the target's token.** Summing $k$ terms, $(1 - \alpha^k)/(1 - \alpha)$,
  counts at most $k$ tokens per round. Every round yields at least one token even
  if all drafts are rejected; that is why speculative decoding never produces fewer
  tokens per target call than plain decoding.
- **Leaving the target call out of the cost.** With cost $k c$ the "speedup" of a
  cheap draft is enormous and grows without bound as $c \to 0$. The target pass is
  the dominant cost of every round.
- **Charging $k + 1$ draft calls.** The bonus token comes from the target's pass,
  not from the draft: $k$ draft calls, one target call.
- **Dividing by zero at $\alpha = 1$.** The closed form is $0/0$ there, and in
  Python that is a `ZeroDivisionError`, not a NaN. A perfect draft is not exotic:
  it is what a copy task looks like.
- **Patching it with an epsilon.** Clamping $\alpha$ to $1 - 10^{-9}$ trades the
  exception for cancellation: $1 - \alpha^{k+1}$ is then a difference of two numbers
  that agree to nine digits, and only about seven correct digits survive. Near 1 the
  term-by-term sum is the stable formula; at exactly 1 the answer is $k + 1$.
- **Never recommending $k = 0$.** A search that starts at $k = 1$ reports the least
  bad draft length when the honest answer is to switch speculation off.
- **Maximizing tokens per round.** $E$ increases with $k$ forever; the quantity to
  maximize is tokens per unit of time.

## Complexity

`expected_tokens` and `speedup` are $O(1)$ (or $O(k)$ with the explicit sum);
`best_k` is $O(k_\text{max})$ evaluations.

## In the wild

The model is from Leviathan, Kalman, and Matias, "Fast Inference from Transformers
via Speculative Decoding" (ICML 2023, arXiv:2211.17192), with their $\gamma$ written
as $k$ here. Their Equation 1 is the expected number of generated tokens,
$(1 - \alpha^{\gamma+1})/(1 - \alpha)$; Theorem 3.8 states that the expected
improvement factor in total wall time is
$(1 - \alpha^{\gamma+1}) / ((1 - \alpha)(\gamma c + 1))$, with the cost coefficient
$c$ defined as the ratio between the time of one draft run and one target run; and
Corollary 3.9 is the $\alpha > c$ condition, with a guaranteed improvement of at
least $(1 + \alpha)/(1 + c)$. They note that the optimal $\gamma$ is found
numerically, as here, and report $\alpha$ between about 0.5 and 0.9 for typical
draft and target pairs, with $c$ below 0.05.

Real acceptances are not independent and identically distributed: $\alpha$ varies
by prompt and by position, and easy stretches alternate with hard ones. Engines
therefore treat the formula as a guide and adapt online. Hugging Face
`transformers` raises the draft length after a fully accepted round and lowers it
after a rejection, and vLLM has offered a switch that disables speculation above a
batch size, when the batch is large enough that the target has no spare capacity and the cost model's premise
fails. With a lookup draft (`c = 0`) the formula says the longest draft is always
best; in practice the extra verified positions are not free either, which caps the
useful length.
