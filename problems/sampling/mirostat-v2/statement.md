Implement one step of Mirostat v2, a stateful sampler that controls the average
surprise (log-perplexity) of generated text. At each step, Mirostat v2 keeps only tokens
whose surprise is at most a running limit $\mu$, samples from those kept tokens, and
updates $\mu$ based on how surprising the drawn token was relative to a target
$\tau$.

## Specification

```python
def mirostat_step(logits, mu, tau, eta, u): ...  # -> (token, new_mu)
```

The input is a batch of $B$ independent sequences over a vocabulary of $V$ tokens.
Each sequence $b$ has its own logit row `logits[b]`, its own running limit `mu[b]`,
and its own uniform random draw `u[b]`. The target surprise `tau` and learning rate
`eta` are Python floats shared across all rows. The function holds no internal
state. Instead, the caller passes each row's `mu` in and feeds the returned `new_mu`
to the next step.

For a single row, let $p = \operatorname{softmax}(\text{logits})$ be the model's
token distribution. The **surprise** of token $i$, measured in **bits**, is

$$
s_i = -\log_2 p_i .
$$

A masked token (logit $-\infty$) has $p_i = 0$ and surprise $+\infty$. Each step
performs three operations on each row.

1. **Truncate.** Keep every token with $s_i \le \mu$. If no token qualifies, keep
   only the single most likely token, breaking ties in favor of the lowest token
   index.
2. **Sample.** Draw a token from the kept set by inverse CDF in token order, as in
   `categorical-sample`. Define unnormalized kept weights $q_i = p_i$ if token $i$
   is kept and $q_i = 0$ otherwise, and let $c_k = \sum_{i \le k} q_i$ be their
   cumulative sums. The total kept probability mass is $c_{V-1}$, and the drawn
   token is

   $$
   \text{token} = \min \{\, k : c_k > u \cdot c_{V-1} \,\}.
   $$

   Scaling $u$ by $c_{V-1}$ is equivalent to renormalizing the kept probabilities
   before sampling.
3. **Update.** Compute the updated limit `new_mu` ($\mu'$) using the surprise of the
   drawn token under the model's original distribution $p$ from before truncation:

   $$
   \mu' = \mu - \eta\,(s_{\text{token}} - \tau).
   $$

Additional parameter rules:

- `logits` entries may be $-\infty$ to represent masked tokens, and every row has at
  least one finite entry. $+\infty$ and NaN never appear.
- `mu` is in bits and may be any finite number, including zero or negative values
  after a run of surprising tokens. When `mu` is below every token's surprise, step
  1 falls back to keeping only the most likely token.
- `tau` satisfies $\tau > 0$ (in bits), and `eta` satisfies $0 < \eta \le 1$.
- `u` satisfies $0 \le u < 1$.
- `token` contains integer token indices in $[0, V)$.

A new sequence starts with $\mu = 2\tau$. Managing that initial value is the
caller's job, so the tests pass `mu = 2 * tau` on the first step of a run and pass
the returned `new_mu` on subsequent steps.

Do not draw random numbers yourself, and do not call your framework's categorical or
multinomial sampler.

> [!NOTE]
> Step 3 uses $-\log_2 p_{\text{token}}$ from the original distribution $p$, not the
> surprise under the renormalized distribution from step 2. This matches the Mirostat
> paper and its authors' reference code. Both llama.cpp and text-generation-webui use
> the renormalized probability instead, and the editorial explains how that changes
> the resulting perplexity.

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

- **Row 1.** Token 3 has probability $1/4$ under the model and $1/3$ among the kept
  tokens. The update uses the original probability (2 bits), not the renormalized
  value ($\log_2 3 \approx 1.58$ bits).
- **Row 2.** Even the most likely token has a surprise of 1 bit, which exceeds
  $\mu = 0.5$, so it is kept alone via the fallback rule. Because 1 bit is below the
  target $\tau = 3.0$, the limit increases.
- **Row 3.** The masked token has infinite surprise and is excluded. The drawn token
  has surprise 3 bits, matching the target, so $\mu$ does not change.

## Constraints

- $1 \le B \le 16$, $1 \le V \le 200$, and all finite logits lie in $[-30, 30]$.
- Returned `token` values are compared exactly, and `new_mu` is compared within
  $10^{-4}$. Float32 rounding cannot flip a test outcome. In every compared step, no
  token's surprise is within $2 \times 10^{-4}$ bits of $\mu$ (except possibly the
  most likely token, which is kept either way), and no cumulative sum is within
  $2 \times 10^{-5}$ of the sampling threshold relative to the kept mass.
- **Long-run cases.** The tests run 16 sequences for 250 steps on a fixed logit row
  (a Zipf-like distribution over 200 tokens with entropy about 5.5 bits), starting at
  $\mu = 2\tau$ and feeding `new_mu` back at each step. Every step is checked against
  the specification with a tolerance of $10^{-3}$. The mean of $s_{\text{token}}$
  over steps 50 to 249 across all sequences must fall within 0.1 bits of $\tau = 3$
  (and within 0.15 bits for $\tau = 4.5$). This band is generous, as the reference
  lands within 0.02 bits.
- No Python loop over rows or tokens is needed, and no sort is needed.

## Why this matters

Top-k and top-p fix the shape of the truncation and let text quality vary with the
model's confidence. On a flat distribution they admit low-quality tokens, and on a
peaked distribution they become repetitive. Mirostat fixes the outcome instead by
targeting the average surprise of the generated text (the log of its perplexity) and
adjusting the truncation threshold automatically.

Mirostat v2 ships in llama.cpp, Ollama, and text-generation-webui as `mirostat = 2`
with parameters `tau` and `eta`. It is also the first stateful sampler in this track,
which introduces common production bugs. Because `mu` belongs to a single sequence,
a server must reset it for each new request, track it per row in a batch, and persist
it when pausing or resuming a generation. Measuring surprise in the wrong unit or
after renormalization still produces fluent text, but at a perplexity the user never
requested.
