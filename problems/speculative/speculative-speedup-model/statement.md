Implement the analytic model that tells you, before running anything, whether
speculative decoding pays off: the expected number of tokens one round produces, the
resulting wall-clock speedup, and the draft length that maximizes it. The formulas
are one line each; the substance is the token the target model always contributes,
the target call in the cost of a round, and the case where every draft is accepted.

## Specification

```python
def expected_tokens(alpha: float, k: int) -> float: ...
def speedup(alpha: float, k: int, c: float) -> float: ...
def best_k(alpha: float, c: float, k_max: int) -> int: ...
```

The model (Leviathan et al., 2023). One *round* of speculative decoding runs the
draft model `k` times to propose `k` tokens, then runs the target model once to
verify them. Each draft token is accepted independently with probability `alpha`,
but only the accepted **prefix** is kept: verification stops at the first rejection.
The target's pass then supplies one more token, the correction at the first
rejection or, if all `k` drafts were accepted, a bonus token. A round therefore
produces between 1 and `k + 1` tokens.

**`expected_tokens(alpha, k)`** is the expected number of tokens a round produces.
The round produces more than $i$ tokens exactly when the first $i$ drafts are all
accepted, which has probability $\alpha^i$, so

$$
E(\alpha, k) \;=\; \sum_{i=0}^{k} \alpha^{i}
\;=\; \frac{1 - \alpha^{k+1}}{1 - \alpha} \quad (\alpha \ne 1),
\qquad
E(1, k) = k + 1 .
$$

**`speedup(alpha, k, c)`** is the expected wall-clock improvement over decoding with
the target model alone. `c` is the cost of one draft call divided by the cost of one
target call. A round costs $k c + 1$ target-call units, and plain decoding produces
exactly one token per unit:

$$
S(\alpha, k, c) \;=\; \frac{E(\alpha, k)}{k c + 1} .
$$

**`best_k(alpha, c, k_max)`** is the draft length $k \in \{0, 1, \dots, k_\text{max}\}$
that maximizes $S(\alpha, k, c)$. $k = 0$ means "do not speculate" and has speedup
exactly 1, so it is the answer whenever no draft length helps. If several $k$ attain
the maximum, return the smallest.

- `alpha`: a float in $[0, 1]$, the acceptance rate, **including exactly `1.0`**.
- `k`, `k_max`: ints $\ge 0$.
- `c`: a float $\ge 0$. `c = 0` models a free draft (a lookup in the prompt); `c`
  may exceed 1.
- `expected_tokens` and `speedup` return a Python `float`; `best_k` returns a Python
  `int`.

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

With an acceptance rate of 0.8 and four draft tokens, a round yields
$1 + 0.8 + 0.8^2 + 0.8^3 + 0.8^4 = 3.3616$ tokens on average. If a draft call costs
5% of a target call, the round costs $4 \times 0.05 + 1 = 1.2$ units, and
$3.3616 / 1.2 \approx 2.80$.

Longer drafts help only up to a point, because the chance of reaching draft $i$
decays like $0.8^i$ while its cost does not:

| `k` | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| $E(0.8, k)$ | 1 | 1.8 | 2.44 | 2.952 | 3.3616 | 3.9514 | 4.3289 | 4.5705 | 4.7251 |
| $S(0.8, k, 0.05)$ | 1 | 1.7143 | 2.2182 | 2.5670 | 2.8013 | 3.0396 | 3.0921 | 3.0470 | 2.9532 |
| $S(0.8, k, 0.3)$ | 1 | 1.3846 | 1.5250 | 1.5537 | 1.5280 | 1.4112 | 1.2732 | 1.1426 | 1.0272 |

With a cheap draft the best length is 8; with a draft that costs 30% of the target
it is 3. With `alpha = 0.3` and `c = 0.5`, one draft token gives
$1.3 / 1.5 \approx 0.87$ and longer drafts are worse: speculation slows decoding
down, and the answer is 0. With `alpha = 1.0` every round yields $k + 1$ tokens.

## Constraints

- `alpha` is either exactly `1.0` or at most `0.999`; $0 \le k, k_\text{max} \le 64$;
  $0 \le c \le 4$.
- Floats are compared with a relative tolerance of `1e-9`. With `alpha` in the
  stated range, both the closed form and the term-by-term sum are accurate enough.
- In every `best_k` test the maximum is unique, and it exceeds the speedup of every
  other draft length by a relative margin of at least `1e-6`, so rounding cannot
  change the answer.

## Why this matters

Speculative decoding is not free speed: it trades draft-model compute for fewer
target passes, and whether the trade wins depends on two measurable numbers, the
acceptance rate and the cost ratio. This model is what an engineer uses to choose a
draft model and its lookahead, and what adaptive engines evaluate online to shorten
the draft when acceptance drops. Its failure modes are the ones in this problem.
Leave out the target call and the model promises unbounded speedups from long
drafts; leave out the bonus token and it undersells short ones; and a production
dashboard that divides by $1 - \alpha$ falls over on the first prompt where the
draft is perfect, typically a copy task, the workload where speculation shines.
