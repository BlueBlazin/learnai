Implement the verification step of Medusa's *typical acceptance*. Instead of requiring each draft token to match the target model's top choice, accept a draft token whenever the target model assigns it a probability above a per-position threshold that relaxes as the target becomes more uncertain.

![This problem is Verify, here a probability threshold in place of an exact match.](assets/map.svg)

> [!WARNING]
> This scheme is **lossy**. Unlike greedy verification or speculative sampling, its output is neither the target model's greedy decode nor an exact sample from the target's distribution. It trades exactness for longer accepted prefixes, controlled by the two threshold hyperparameters.

## Specification

```python
def typical_accept(p, draft_tokens, epsilon, delta): ...  # -> (n_accepted, next_token)
```

Suppose the target model has been run once on the verified prefix followed by `K` draft tokens over a vocabulary of size `V`.

- `p` contains the `K + 1` relevant next-token **probability** distributions from the target model, aligned as in `greedy-speculative-verify`. Row `i` for $0 \le i < K$ is the target's distribution at the position of `draft_tokens[i]` (conditioned on the verified prefix and `draft_tokens[0..i-1]`), and row `K` is the distribution after the last draft token. Each row is non-negative and sums to 1 within float32 rounding. Entries may be exactly `0` (for masked tokens), and a row may be one-hot.
- `draft_tokens` holds the `K` proposed token IDs.
- `epsilon` is a Python `float` in $(0, 1]$ giving the hard probability threshold $\epsilon$.
- `delta` is a Python `float` in $(0, 1]$ giving the entropy-scaled threshold multiplier $\delta$.

For each row $i$, measure the target model's uncertainty with its Shannon entropy $H_i$ in **nats** (using the convention $0 \ln 0 = 0$), where $p_{i,v}$ is the probability of token $v$ in row $i$. Then compute the row's acceptance threshold $\tau_i$ as the smaller of $\epsilon$ and $\delta e^{-H_i}$:

$$
H_i = -\sum_{v} p_{i,v} \ln p_{i,v},
\qquad
\tau_i = \min\!\big(\epsilon,\ \delta \, e^{-H_i}\big).
$$

Draft token $i$ **passes** when its target probability is strictly greater than its row's threshold:

$$
p_{i,\,\texttt{draft\_tokens}[i]} \;>\; \tau_i .
$$

Let $n$ (returned as `n_accepted`) be the length of the longest prefix of draft tokens that all pass, and take the target model's greedy token from the row immediately following that prefix:

$$
n = \min\big(\{\, i : 0 \le i < K,\ \text{draft token } i \text{ does not pass} \,\} \cup \{K\}\big),
\qquad
\texttt{next\_token} = \operatorname{argmax}_v\; p_{n,v} .
$$

Tokens after the first failure are discarded even if they would pass, because their target distributions were conditioned on a rejected token. The returned `next_token` is the target's greedy replacement at the first failure, or its bonus token after a full acceptance. No threshold applies to `next_token`, which guarantees that every round produces at least one token.

Return `(n_accepted, next_token)` as two Python `int`s, with $0 \le \texttt{n\_accepted} \le K$.

Note two edge cases. A draft token with probability `0` never passes because every threshold $\tau_i$ is strictly positive. A one-hot row has $H_i = 0$ and therefore threshold $\min(\epsilon, \delta)$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(K + 1, V)` | float32 |
| `draft_tokens` | `(K,)` | integer |
| result | `(n_accepted, next_token)` | Python `int`, Python `int` |

## Example

```python
p = [
    [0.70, 0.10, 0.10, 0.10],  # H = 0.940   tau = min(0.09, 0.1171) = 0.09     argmax 0
    [0.25, 0.25, 0.25, 0.25],  # H = 1.386   tau = min(0.09, 0.0750) = 0.075    (tie)
    [0.40, 0.30, 0.24, 0.06],  # H = 1.239   tau = min(0.09, 0.0869) = 0.0869   argmax 0
    [0.05, 0.05, 0.85, 0.05],
]  # row K: only its argmax, 2, can matter

typical_accept(p, [1, 3, 3], epsilon=0.09, delta=0.3)  # (2, 0)
typical_accept(p, [1, 3, 2], epsilon=0.09, delta=0.3)  # (3, 2)
typical_accept(p, [1, 3, 3], epsilon=0.3, delta=0.3)  # (0, 0)
```

**First call.** In row 0, token 1 has probability $0.10$, which is above $\tau_0 = 0.09$. It passes even though the target's top choice is token 0. Row 1 is uniform with entropy $\ln 4$, so the entropy term $0.3 \cdot e^{-\ln 4} = 0.075$ is smaller than $\epsilon$, and token 3 ($p = 0.25$) passes easily. In row 2, the entropy term is $0.3 \cdot e^{-1.239} = 0.0869$, and token 3 ($p = 0.06$) fails. Two tokens are accepted, and row 2 supplies the target's greedy token 0.

**Second call.** The third draft token is now token 2, which has probability $0.24$ in row 2. All three draft tokens pass, and row 3 supplies the bonus token 2.

**Third call.** With $\epsilon = 0.3$, row 0's threshold becomes $\min(0.3, 0.1171) = 0.1171$, which token 1 ($0.10$) fails. Zero tokens are accepted, positions 1 and 2 are ignored, and row 0's argmax (token 0) is returned.

Row 1 has four equal probabilities, which is harmless here because it never supplies `next_token`. No test takes the argmax of a row with a tie.

## Constraints

- $1 \le K \le 16$ and $2 \le V \le 64$.
- Outputs are compared exactly. In every test, each draft token's probability differs from its row's threshold by a relative margin of at least `1e-3`, and the top two entries of the row supplying `next_token` differ by at least `1e-3`, so float32 rounding cannot change the result. The one exception tests the strict inequality, where the active threshold is $\epsilon$ and both $\epsilon$ and the draft token's probability are the exact same power of two.
- A few vectorized tensor operations suffice, with no Python loop over tokens needed.

## Why this matters

Exact verification loses speed at higher sampling temperatures. When the target distribution is spread across many plausible continuations, insisting on reproducing one exact choice rejects drafts that the target model considers perfectly reasonable. Medusa's typical acceptance relaxes the check by keeping any draft token that is *typical* under the target distribution, lowering the probability bar further when the target is uncertain.

Because typical acceptance alters the generated output, every detail of the rule affects model quality. Computing entropy in bits instead of nats, taking `max` instead of `min`, or counting passes after the first failure all produce fluent text with a higher acceptance rate. On a latency dashboard, those bugs look like a free speedup while silently degrading output quality.
