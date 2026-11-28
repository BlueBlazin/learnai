Implement the analytic performance model for speculative decoding. You will compute the expected number of tokens produced per round, the resulting wall-clock speedup over standard decoding, and the draft length that maximizes that speedup.

## Specification

```python
def expected_tokens(alpha: float, k: int) -> float: ...
def speedup(alpha: float, k: int, c: float) -> float: ...
def best_k(alpha: float, c: float, k_max: int) -> int: ...
```

This problem implements the model from Leviathan et al. (2023). In one *round* of speculative decoding, the draft model runs `k` times to propose `k` draft tokens, and the target model runs once to verify them. Each draft token is accepted independently with probability `alpha` ($\alpha$). Verification stops at the first rejection, so only the accepted **prefix** is kept. The target pass then always contributes one more token: either the correction at the first rejection, or a bonus token if all `k` drafts were accepted. A round therefore produces between 1 and `k + 1` tokens.

**`expected_tokens(alpha, k)`** returns the expected number of tokens produced in one round. A round produces more than $i$ tokens exactly when the first $i$ draft tokens are all accepted, which happens with probability $\alpha^i$. Summing over $i = 0, \dots, k$ gives

$$
E(\alpha, k) = \sum_{i=0}^{k} \alpha^{i}
= \frac{1 - \alpha^{k+1}}{1 - \alpha} \quad (\alpha \ne 1),
\qquad
E(1, k) = k + 1.
$$

**`speedup(alpha, k, c)`** returns the expected wall-clock speedup over decoding with the target model alone. Here `c` is the cost of one draft call divided by the cost of one target call. One round costs $k c + 1$ target-call units, while plain decoding produces 1 token per target-call unit:

$$
S(\alpha, k, c) = \frac{E(\alpha, k)}{k c + 1}.
$$

**`best_k(alpha, c, k_max)`** returns the draft length $k \in \{0, 1, \dots, k_\text{max}\}$ that maximizes $S(\alpha, k, c)$. Setting $k = 0$ disables speculation and yields a speedup of exactly 1, so `best_k` returns `0` whenever no positive draft length improves over plain decoding. If multiple values of $k$ tie for the maximum, return the smallest.

- `alpha`: Python `float` in $[0, 1]$, the per-token acceptance rate, **including exactly `1.0`**.
- `k`, `k_max`: Python `int` values $\ge 0$.
- `c`: Python `float` $\ge 0$, the draft-to-target cost ratio (`c = 0` models a free draft such as prompt lookup; `c` may exceed 1).
- `expected_tokens` and `speedup` return a Python `float`. `best_k` returns a Python `int`.

## Example

```python
expected_tokens(0.8, 4)        # 3.3616
speedup(0.8, 4, 0.05)          # 2.8013333333333335
best_k(0.8, 0.05, 20)          # 8      (speedup 3.0921)
best_k(0.8, 0.3, 20)           # 3      (speedup 1.5537)
best_k(0.3, 0.5, 20)           # 0
expected_tokens(1.0, 4)        # 5.0
speedup(1.0, 4, 0.05)          # 4.166666666666667
```

With `alpha = 0.8` and `k = 4`, a round produces $1 + 0.8 + 0.8^2 + 0.8^3 + 0.8^4 = 3.3616$ tokens on average. When each draft call costs 5% of a target call (`c = 0.05`), the round costs $4 \times 0.05 + 1 = 1.2$ target-call units, giving a speedup of $3.3616 / 1.2 \approx 2.80$.

Longer drafts help only up to a point, because the probability of reaching draft $i$ decays as $0.8^i$ while its cost stays constant:

| `k` | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| $E(0.8, k)$ | 1 | 1.8 | 2.44 | 2.952 | 3.3616 | 3.9514 | 4.3289 | 4.5705 | 4.7251 |
| $S(0.8, k, 0.05)$ | 1 | 1.7143 | 2.2182 | 2.5670 | 2.8013 | 3.0396 | 3.0921 | 3.0470 | 2.9532 |
| $S(0.8, k, 0.3)$ | 1 | 1.3846 | 1.5250 | 1.5537 | 1.5280 | 1.4112 | 1.2732 | 1.1426 | 1.0272 |

With a cheap draft (`c = 0.05`), the optimal length is `k = 8`. With a slower draft (`c = 0.3`), it drops to `k = 3`. When `alpha = 0.3` and `c = 0.5`, even one draft token gives $1.3 / 1.5 \approx 0.87 < 1$ and longer drafts perform worse, so `best_k` returns `0`. When `alpha = 1.0`, every round yields `k + 1` tokens.

## Constraints

- `alpha` is either exactly `1.0` or at most `0.999`.
- $0 \le k, k_\text{max} \le 64$ and $0 \le c \le 4$.
- Floating-point outputs are compared with a relative tolerance of `1e-9`. Within the stated range of `alpha`, both the closed form and the explicit sum meet this accuracy.
- In every `best_k` test, the maximum is unique and exceeds every other draft length's speedup by a relative margin of at least `1e-6`, so floating-point rounding cannot change the answer.

## Why this matters

Speculative decoding trades draft-model compute for fewer target passes. Whether that trade pays off depends on two measurable quantities: the acceptance rate `alpha` and the cost ratio `c`. Engineers use this model to pick a draft model and lookahead `k`, and adaptive inference engines evaluate it online to shorten drafts when acceptance drops.

Each edge case in this problem reflects a real bug. Omitting the target call from the cost predicts unbounded speedups from long drafts, omitting the bonus token underestimates short drafts, and dividing blindly by $1 - \alpha$ crashes on the first prompt with perfect acceptance. That case is typically a copy-heavy task where speculation shines most.
