Implement one step of Mirostat v2, a sampler with state: it keeps a running limit
$\mu$ on how surprising the next token may be, removes every token above the limit,
samples from the rest, and then moves the limit toward a target. The truncation is
one comparison; what decides whether the feedback loop holds the target is which
surprise goes into the update, and in which unit.

## Specification

```python
def mirostat_step(logits, mu, tau, eta, u): ...  # -> (token, new_mu)
```

Each row is an independent sequence with its own limit `mu[b]`. Randomness is an
input: row $b$ is sampled with its own uniform number `u[b]`. The function holds no
state; the caller passes each row's `mu` in and feeds the returned one to the next
step.

For one row, let $p = \operatorname{softmax}(\text{logits})$. The **surprise** of
token $i$, in **bits**, is

$$
s_i = -\log_2 p_i ,
$$

which is $+\infty$ for a masked token. One step does three things.

1. **Truncate.** Keep the tokens with $s_i \le \mu$. If no token qualifies, keep the
   single most likely token (the lowest index if several share the largest logit).
2. **Sample** from the kept tokens by inverse CDF in token order, as in
   `categorical-sample`. With $q_i = p_i$ for a kept token and $q_i = 0$ otherwise,
   and cumulative sums $c_k = \sum_{i \le k} q_i$,

   $$
   \text{token} = \min \{\, k : c_k > u \cdot c_{V-1} \,\}.
   $$

   Scaling $u$ by the kept mass $c_{V-1}$ is the renormalization of the truncated
   distribution.
3. **Update** the limit with the surprise of the drawn token under the model's
   distribution $p$, the one from before the truncation:

   $$
   \mu' = \mu - \eta\,(s_{\text{token}} - \tau).
   $$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked
  token); every row has at least one finite entry. `+inf` and NaN never appear.
- `mu`: a float32 tensor of shape `(B,)`, in bits. It may be any finite number,
  also zero or negative (after a run of surprising tokens); step 1 then keeps only
  the most likely token.
- `tau`: a Python float, the target surprise in bits, $\tau > 0$. `eta`: a Python
  float, the learning rate, $0 < \eta \le 1$. Both are the same for every row.
- `u`: a float32 tensor of shape `(B,)` with $0 \le u < 1$.
- Returns a tuple `(token, new_mu)`: an integer tensor of shape `(B,)` with values
  in $[0, V)$, and a float32 tensor of shape `(B,)`.

A sequence starts with $\mu = 2\tau$. That is the caller's job: the tests pass
`mu = 2 * tau` on the first step of a run and the returned `new_mu` afterwards.

Do not draw random numbers yourself, and do not call your framework's categorical or
multinomial sampler.

> [!NOTE]
> The surprise in step 3 is $-\log_2 p_{\text{token}}$, not the surprise under the
> renormalized distribution of step 2. This follows the paper and its authors' code.
> llama.cpp and text-generation-webui use the renormalized probability instead; the
> editorial shows what that changes.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `mu` | `(B,)` | float32 |
| `u` | `(B,)` | float32 |
| `token` | `(B,)` | integer |
| `new_mu` | `(B,)` | float32 |

## Example

```python
row = log([2, 8, 1, 4, 1])  # p = [1/8, 1/2, 1/16, 1/4, 1/16], surprise [3, 1, 4, 2, 4] bits
logits = [row, row, row, log([2, 8, 0, 4, 2])]  # last row: token 2 is masked (-inf)
mu = [6.0, 2.5, 0.5, 3.5]
u = [0.9, 0.7, 0.99, 0.95]

mirostat_step(logits, mu, tau=3.0, eta=0.5, u=u)  # ([3, 3, 1, 4], [6.5, 3.0, 1.5, 3.5])
```

| row | $\mu$ | kept tokens | cumulative sums $c$ | $u \cdot c_{V-1}$ | token | $s_{\text{token}}$ | $\mu'$ |
|---|---|---|---|---|---|---|---|
| 0 | 6.0 | all | $[0.125, 0.625, 0.6875, 0.9375, 1]$ | 0.9 | 3 | 2 | $6 - 0.5\,(2 - 3) = 6.5$ |
| 1 | 2.5 | 1, 3 | $[0, 0.5, 0.5, 0.75, 0.75]$ | 0.525 | 3 | 2 | $2.5 - 0.5\,(2 - 3) = 3.0$ |
| 2 | 0.5 | 1 (fallback) | $[0, 0.5, 0.5, 0.5, 0.5]$ | 0.495 | 1 | 1 | $0.5 - 0.5\,(1 - 3) = 1.5$ |
| 3 | 3.5 | 0, 1, 3, 4 | $[0.125, 0.625, 0.625, 0.875, 1]$ | 0.95 | 4 | 3 | $3.5 - 0.5\,(3 - 3) = 3.5$ |

Row 1: token 3 has probability $1/4$ under the model and $1/3$ among the kept
tokens. The update uses the former, 2 bits, not $\log_2 3 = 1.58$ bits.

Row 2: even the most likely token has a surprise of 1 bit, above $\mu = 0.5$, so it
is kept alone. It is less surprising than the target, and the limit rises.

Row 3: the masked token has infinite surprise and is never kept. The drawn token
has exactly the target surprise, and the limit does not move.

## Constraints

- $1 \le B \le 16$, $1 \le V \le 200$. Finite logits lie in $[-30, 30]$.
- Tokens are compared exactly, `new_mu` within $10^{-4}$. Float32 rounding cannot
  decide a test: in every compared step, no token's surprise is within
  $2 \times 10^{-4}$ bits of $\mu$ (except the most likely token, which is kept
  either way), and no cumulative sum is within $2 \times 10^{-5}$ of the threshold,
  relative to the kept mass.
- **Long-run cases.** 16 sequences are run for 250 steps on one fixed row of logits
  (a Zipf-like distribution over 200 tokens, entropy about 5.5 bits), starting at
  $\mu = 2\tau$ and feeding `new_mu` back. Every step is checked against the
  definition with a tolerance of $10^{-3}$, and the mean of $s_{\text{token}}$ over
  steps 50 to 249 of all sequences must be within 0.1 bits of $\tau = 3$ (0.15 bits
  for $\tau = 4.5$). The band is wide: the reference lands within 0.02 bits.
- No Python loop over rows or tokens is needed, and no sort.

## Why this matters

Top-k and top-p fix the shape of the truncation and let the quality of the text
follow from it: on a flat distribution they admit junk, on a peaked one they repeat.
Mirostat fixes the outcome instead, the average surprise of the generated text (the
logarithm of its perplexity), and lets the truncation move. It ships in llama.cpp,
Ollama, and text-generation-webui as `mirostat = 2` with `tau` and `eta`. It is also
the first sampler in this track with state, which is where its production bugs
come from: `mu` belongs to a sequence, so it has to be reset when a new one starts,
kept per row in a batch, and carried through a cached or resumed generation. A
sampler that measures surprise in the wrong unit or after renormalization still
produces fluent text, only at a perplexity nobody asked for.
