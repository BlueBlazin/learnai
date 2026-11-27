## The idea

Model each draft token as accepted independently with probability $\alpha$. Because verification keeps only the prefix before the first rejection, the number of accepted draft tokens $n$ follows a geometric distribution capped at $k$, with $P(n \ge i) = \alpha^i$ for $0 \le i \le k$. The target's single pass always yields one valid token of its own—either the correction at the first rejection or a bonus token after $k$ acceptances—so the round emits $n + 1$ tokens. Applying the tail-sum formula gives

$$
E(\alpha, k) = \mathbb{E}[n + 1] = 1 + \sum_{i=1}^{k} P(n \ge i)
= \sum_{i=0}^{k} \alpha^i = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

To compute wall-clock speedup, measure time in target calls. Plain decoding produces 1 token per target call. One speculative round pays for $k$ draft calls at cost $c$ each plus 1 target call, and produces $E(\alpha, k)$ tokens:

$$
S(\alpha, k, c) = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(k c + 1)}.
$$

This model relies on the core hardware assumption behind speculative decoding: scoring $k + 1$ positions in a single batched target pass takes about the same time as scoring one position when the accelerator has spare parallel capacity.

As $k$ grows, the numerator saturates at $1 / (1 - \alpha)$ while the denominator grows linearly in $k$. Therefore $S(\alpha, k, c)$ rises, peaks, and eventually falls, so we can find the optimal integer $k$ by direct search. Evaluating $k = 1$ reveals when speculation helps at all: $S = (1 + \alpha) / (1 + c) > 1$ if and only if $\alpha > c$. When $\alpha \le c$, every additional term $\alpha^i \le c$ adds less to the numerator than $c$ adds to the denominator, so no positive $k$ improves over plain decoding and `best_k` returns `0`.

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

Python's `max` returns the first element that attains the maximum, which breaks ties in favor of the smallest `k`. Including `k = 0` in `range(k_max + 1)` naturally yields $E = 1$ and $S = 1$ without a separate branch.

## Common mistakes

- **Forgetting the target's token.** Summing only $k$ terms, $(1 - \alpha^k) / (1 - \alpha)$, caps the output at $k$ tokens per round. Every round produces at least one token even when all drafts are rejected, which is why speculative decoding never yields fewer tokens per target call than plain decoding.
- **Omitting the target call from the cost.** Dividing by $k c$ instead of $k c + 1$ produces an unbounded speedup as $c \to 0$. The target pass is the dominant cost of every round.
- **Charging `k + 1` draft calls.** The bonus token comes from the target pass, not an extra draft pass. A round makes `k` draft calls and 1 target call.
- **Dividing by zero at $\alpha = 1$.** The closed form evaluates to `0 / 0`, raising a `ZeroDivisionError` in Python. Perfect acceptance is common on copy-heavy prompts and must return `float(k + 1)`.
- **Patching $\alpha = 1$ with an epsilon.** Clamping $\alpha$ to $1 - 10^{-9}$ avoids the exception but causes catastrophic cancellation in $1 - \alpha^{k+1}$, leaving only about seven accurate digits. Near 1, the explicit sum is numerically stable, and at exactly 1 the result is `k + 1`.
- **Never recommending `k = 0`.** Starting the search at `k = 1` forces speculation even when it slows decoding down.
- **Maximizing tokens per round instead of speedup.** $E(\alpha, k)$ increases monotonically with $k$, so optimizing $E$ always picks `k_max` regardless of draft cost.

## Complexity

`expected_tokens` and `speedup` run in $O(1)$ time (or $O(k)$ if using the explicit sum). `best_k` evaluates `speedup` $k_\text{max} + 1$ times, taking $O(k_\text{max})$ time and $O(1)$ extra memory.

## In the wild

This model comes from Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192), where draft length is denoted $\gamma$ rather than $k$. Their Equation 1 gives the expected tokens $(1 - \alpha^{\gamma+1}) / (1 - \alpha)$. Theorem 3.8 gives the expected wall-time improvement factor $(1 - \alpha^{\gamma+1}) / ((1 - \alpha)(\gamma c + 1))$, with $c$ defined as the ratio of one draft call to one target call. Corollary 3.9 proves the $\alpha > c$ threshold and shows a guaranteed speedup of at least $(1 + \alpha) / (1 + c)$. They find the optimal $\gamma$ numerically as we do here, reporting typical values of $\alpha$ between 0.5 and 0.9 with $c < 0.05$.

In practice, token acceptances are not independent and identically distributed. Acceptance varies across prompts and positions as easy spans alternate with hard ones, so serving engines use this model as a baseline and adapt online. Hugging Face `transformers` increases the draft length after a fully accepted round and decreases it after a rejection. vLLM provides a batch-size threshold that disables speculation when large batches saturate the target model's compute and violate the free-verification assumption. Similarly, for a free lookup draft (`c = 0`), the formula favors making `k` as large as possible, whereas real hardware still pays a non-zero cost for verifying extra positions.
