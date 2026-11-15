Implement the verification step of Medusa's *typical acceptance*: a draft token is
kept when the target model finds it plausible enough, not only when it is the token
the target would have chosen itself. The bar is a probability threshold that drops
when the target is uncertain. The step is a handful of tensor operations; the
substance is the entropy (in nats, with zero probabilities), the `min`, the strict
comparison, and stopping at the first failure.

> [!WARNING]
> This scheme is **lossy**. Unlike greedy verification or speculative sampling, its
> output is neither the target model's greedy decode nor a sample from the target's
> distribution. It trades exactness for longer accepted runs, and the two thresholds
> set how much.

## Specification

```python
def typical_accept(p, draft_tokens, epsilon, delta): ...  # -> (n_accepted, next_token)
```

The target model was run once on the verified sequence followed by the `K` draft
tokens. `p` holds the `K + 1` next-token **probability** distributions that matter,
aligned as in `greedy-speculative-verify`:

- row `i`, for $0 \le i < K$, is the target's distribution at **the position of
  `draft_tokens[i]`**, given the verified sequence and `draft_tokens[0..i-1]`;
- row `K` is the distribution after the last draft token.

For every row, let $H_i$ be its entropy in **nats**, with the convention
$0 \ln 0 = 0$, and $\tau_i$ its threshold:

$$
H_i = -\sum_{v} p_{i,v} \ln p_{i,v},
\qquad
\tau_i = \min\!\big(\epsilon,\ \delta \, e^{-H_i}\big).
$$

Draft token $i$ **passes** when its probability is strictly above the threshold of
its row:

$$
p_{i,\,\texttt{draft\_tokens}[i]} \;>\; \tau_i .
$$

Then

$$
n = \min\big(\{\, i : 0 \le i < K,\ \text{draft token } i \text{ does not pass} \,\} \cup \{K\}\big),
\qquad
\texttt{next\_token} = \operatorname{argmax}_v\; p_{n,v} .
$$

In words: accept the longest prefix of the draft in which every token passes. Tokens
after the first failure are discarded even if they would pass, because their rows
were computed from a context that contains the rejected token. The token that
follows the accepted prefix is the target's own **greedy** token for that position:
the replacement at the first failure, or the bonus token after a full acceptance. It
is taken unconditionally (no threshold applies to it), which is what guarantees that
every round produces at least one token.

- `p`: float32 tensor of shape `(K + 1, V)`. Every row is a probability distribution
  (non-negative, summing to 1 within float32 rounding). Entries may be exactly `0`
  (masked tokens), and a row may be one-hot.
- `draft_tokens`: integer tensor of shape `(K,)`.
- `epsilon`: a Python float in $(0, 1]$, the hard threshold $\epsilon$.
- `delta`: a Python float in $(0, 1]$, the entropy-dependent threshold $\delta$.
- Returns `(n_accepted, next_token)` as two Python `int`s, with
  $0 \le \texttt{n\_accepted} \le K$.

A token of probability `0` never passes, since every threshold is positive. A
one-hot row has $H = 0$ and threshold $\min(\epsilon, \delta)$.

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

**First call.** Token 1 has probability $0.10$ in row 0, above $\tau_0 = 0.09$: it
passes, although the target's own choice would have been token 0. Row 1 is uniform,
so its entropy is $\ln 4$ and the entropy term, $0.3 \cdot e^{-\ln 4} = 0.075$, is
the smaller one: token 3, with probability $0.25$, passes easily. In row 2 the
entropy term is $0.3 \cdot e^{-1.239} = 0.0869$, and token 3 has probability $0.06$:
it fails. Two tokens are accepted, and the target's greedy token of row 2, token 0,
comes next.

**Second call.** The last draft token is now token 2, with probability $0.24$ in
row 2. All three pass, and row 3 supplies the bonus token 2.

**Third call.** With $\epsilon = 0.3$ the threshold of row 0 is
$\min(0.3, 0.1171) = 0.1171$, and $0.10$ is below it. Nothing is accepted; the
tokens at positions 1 and 2 are not looked at, and the next token is row 0's argmax.

Row 1 has four equal entries, which is harmless: it never supplies `next_token` in
these calls, and no test takes the argmax of a row with a tie.

## Constraints

- $1 \le K \le 16$ and $2 \le V \le 64$.
- Results are compared exactly. In every test, the probability of each draft token
  differs from the threshold of its row by a relative margin of at least `1e-3`, and
  the largest and second-largest entry of the row that supplies `next_token` differ
  by at least `1e-3`, so float32 rounding cannot change the answer. The one exception is a test of the
  strict inequality: there the threshold is $\epsilon$, and $\epsilon$ and the
  probability are the same power of two, exactly.
- A handful of tensor operations suffice; no Python loop over tokens is needed.

## Why this matters

Exact verification gets slower as sampling gets more creative: at a temperature
above zero the target could have produced many continuations, and a scheme that
insists on reproducing one particular choice throws away drafts the target would
have been perfectly happy with. Medusa's typical acceptance answers with a relaxed
rule: accept what is *typical* for the target, measured by its own probability, and
relax further where its distribution is flat. Because the scheme
changes what the model says, its details are quality knobs, not implementation
freedom. Entropy in bits, a `max` where the `min` belongs, or a comparison that
keeps counting after a failure each still produce fluent text and a higher
acceptance rate, which looks like a speedup on a dashboard and is a silent drop in
output quality.
