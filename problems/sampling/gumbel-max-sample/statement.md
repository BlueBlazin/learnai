Draw one token per row from a batch of logits with the Gumbel-max trick: add
independent Gumbel noise to every logit and take the argmax. No softmax, no cumulative
sum, and the winner is distributed exactly as the softmax of the logits. The noise is
made from uniform numbers that the caller supplies, and the care goes into the two
ends of the unit interval, where the transformation blows up.

## Specification

```python
def gumbel_sample(logits, uniforms): ...
```

Randomness is an input: the function is deterministic, and entry `[b, v]` of `logits`
gets its own uniform number `uniforms[b, v]`.

For each entry, clamp the uniform number into the open unit interval and turn it into
standard Gumbel noise:

$$
\tilde u = \min\big(\max(u,\ u_{\min}),\ u_{\max}\big), \qquad
g = -\log(-\log \tilde u),
$$

with $u_{\min} = 1.1754944 \times 10^{-38}$ (the smallest positive normal float32)
and $u_{\max} = 0.99999994 = 1 - 2^{-24}$ (the largest float32 below 1). The token of
row $b$ is

$$
\text{token}_b = \operatorname*{argmax}_v \big( \text{logits}[b, v] + g[b, v] \big).
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked
  token); every row has at least one finite entry. `+inf` and NaN never appear. The
  logits are unnormalized: no softmax is applied before or inside this function.
- `uniforms`: a float32 tensor of shape `(B, V)` with $0 \le u \le 1$. **Both ends
  occur**: a float32 generator returns exactly 0 once in $2^{24}$ draws, and uniform
  numbers computed in float64 and stored as float32 round up to exactly 1.
- Returns an integer tensor of shape `(B,)` with values in $[0, V)$.

Things that follow from the definition, and that the tests check:

- **A masked token is never returned.** The clamp keeps the noise finite,
  $-4.47 \le g \le 16.64$, so a logit of `-inf` has a score of `-inf` whatever its
  uniform number is, also for `u = 1`, and a finite logit has a finite score, also for
  `u = 0`.
- **The result is never NaN-driven.** Without the upper clamp, `u = 1` gives
  $g = +\infty$, and on a masked entry $-\infty + \infty$ is NaN.
- With `uniforms` independent and uniform on $[0, 1)$, token $v$ is returned with
  probability $\operatorname{softmax}(\text{logits}[b])_v$.

Do not draw random numbers yourself, and do not call your framework's categorical,
multinomial, or Gumbel sampler.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `uniforms` | `(B, V)` | float32 |
| result | `(B,)` | integer |

## Example

```python
logits = [[1.0, 2.0, 0.0, -inf], [0.0, 0.0, 0.0, 0.0], [-inf, 3.0, -inf, -inf]]
uniforms = [[0.9, 0.3, 0.5, 0.99], [0.2, 0.6, 0.95, 0.5], [0.7, 0.0, 1.0, 0.5]]

gumbel_sample(logits, uniforms)  # [0, 2, 1]
```

| row | noise $g$ | score = logit + $g$ | token |
|---|---|---|---|
| 0 | $[2.2504, -0.1856, 0.3665, 4.6002]$ | $[3.2504, 1.8144, 0.3665, -\infty]$ | 0 |
| 1 | $[-0.4759, 0.6717, 2.9702, 0.3665]$ | $[-0.4759, 0.6717, 2.9702, 0.3665]$ | 2 |
| 2 | $[1.0309, -4.4698, 16.6355, 0.3665]$ | $[-\infty, -1.4698, -\infty, -\infty]$ | 1 |

Row 0: token 1 has the largest logit, but token 0 drew more noise and wins; that is
the sampling. Token 3 drew the most noise of all and is masked, so it cannot win.

Row 1: with equal logits the largest uniform number wins.

Row 2: the only unmasked token has `u = 0`, which is clamped to $u_{\min}$ and gives
the smallest possible noise, $-\log(87.34) = -4.47$; its score is still finite. Token
2 has `u = 1`, which is clamped to $u_{\max}$ and gives the largest possible noise,
$16.64$; its score is still `-inf`. Unclamped, the scores would be
`[-inf, -inf, nan, -inf]`.

## Constraints

- $1 \le B \le 20000$, $1 \le V \le 1000$, $0 \le u \le 1$. Finite logits lie in
  $[-1000, 1000]$.
- Tokens are compared with the reference exactly. Float32 rounding cannot decide a
  test: in every row the best score exceeds the second best by at least $10^{-3}$, and
  by at least 1 in rows where the best or second-best token has a uniform number of
  exactly 0 or 1.
- **Statistical cases.** The function is called once with 20,000 identical rows of
  logits and independent uniforms, and the empirical distribution of the tokens is
  compared with the softmax of the row using a finite-sample (Chernoff) bound. A
  correct solution fails with probability below $10^{-9}$, and since the seeds are
  fixed, it either always passes or always fails.
- No Python loop over rows or tokens is needed.

## Why this matters

`jax.random.categorical` and `mx.random.categorical` are this function, and so is
every sampler that must run inside a compiled graph or on a GPU: an argmax over
independent per-entry noise parallelizes trivially, needs no sort and no sequential
cumulative sum, and takes logits as they are, masked entries included. The same trick
underlies Gumbel-softmax, Gumbel top-k (sampling without replacement), and several
watermarking schemes. Its numerical edges are where it fails in production: one decoding
step at batch size 128 over a 150,000-token vocabulary consumes more than $2^{24}$
uniform numbers, so a float32 generator is expected to return an exact 0 about once
per step, and a single NaN score makes an argmax return a token that a grammar had
masked.
