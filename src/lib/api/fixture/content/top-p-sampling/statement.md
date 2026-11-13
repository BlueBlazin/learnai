Nucleus sampling draws the next token from the smallest set of tokens whose total probability is at least $p$. It cuts the unreliable tail of the distribution while adapting the cutoff to how confident the model is.

Let $\pi$ sort the vocabulary by descending probability, and let

$$
k^{*} = \min \Bigl\{ k : \sum_{r=1}^{k} P_{\pi(r)} \ge p \Bigr\}.
$$

The nucleus is $\{\pi(1), \dots, \pi(k^{*})\}$. Renormalize over it and sample.

## Randomness is an input

Your function never calls a random number generator. It receives $u \in [0, 1)$ and must return the token selected by inverse-CDF sampling over the nucleus **in sorted order**:

$$
\text{token} = \pi\bigl(\min \{\, r : \textstyle\sum_{s \le r} \tilde P_{\pi(s)} > u \,\}\bigr),
\qquad \tilde P = \frac{P}{\sum_{r \le k^{*}} P_{\pi(r)}} .
$$

This makes the result exactly checkable in every framework.

## Interface

```python
def sample_top_p(logits: Tensor, p: float, u: Tensor) -> Tensor:
    """
    logits: (B, V) float32.    p in (0, 1].    u: (B,) uniforms in [0, 1).
    Returns (B,) int64 token ids.
    """
```

### Example 1

Probabilities $[0.5, 0.3, 0.15, 0.05]$ with $p = 0.9$. The cumulative sums are $[0.5, 0.8, 0.95, 1.0]$, so $k^{*} = 3$ and the nucleus is tokens $\{0, 1, 2\}$ with renormalized mass $[0.526, 0.316, 0.158]$.

| $u$ | Renormalized CDF crossed | Token |
|---:|---|---:|
| $0.10$ | $0.526$ | 0 |
| $0.60$ | $0.842$ | 1 |
| $0.90$ | $1.000$ | 2 |

Token 3 can never be returned.

### Example 2

With $p = 1$ the nucleus is the whole vocabulary and the function is plain inverse-CDF sampling.

## How it is judged

The statistical judge calls your function with $200{,}000$ fixed uniforms per distribution and compares the empirical frequencies to the exact nucleus distribution with a chi-squared test at a false-failure rate below $10^{-9}$. A second set of cases checks boundary uniforms exactly.

> [!WARNING]
> The token that **crosses** the threshold belongs to the nucleus. Dropping it is the most common bug: the output still looks plausible, and only the frequency test catches it.

## Constraints

- $1 \le B \le 64$, $2 \le V \le 50{,}000$.
- Ties in probability are broken by lower token id first (a stable sort).
