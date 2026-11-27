Implement the Gumbel-max trick to sample one token per row from a batch of raw logits. Add independent Gumbel noise derived from caller-supplied uniform numbers to every logit, then take the argmax in each row.

![This problem is Softmax and Sample, done together as one argmax over noisy logits.](assets/map.svg)

## Specification

```python
def gumbel_sample(logits, uniforms): ...
```

The input `logits` is a 2D tensor with one row per sequence in a batch of size `B` and one column per token in a vocabulary of size `V`. Entries may be `-inf` to indicate masked tokens, and every row has at least one finite entry. Neither `+inf` nor NaN appears. The logits are unnormalized, so no softmax is applied before or inside this function.

Randomness is passed in as `uniforms`, making the function deterministic. Entry `uniforms[b, v]` is a float32 value $u \in [0, 1]$ paired with `logits[b, v]`. Both endpoints `0.0` and `1.0` occur in practice. A float32 generator returns `0.0` about once in $2^{24}$ draws, and uniform values generated in float64 can round up to `1.0` when cast to float32.

For each entry, clamp $u$ into the open unit interval and convert it into standard Gumbel noise $g$:

$$
\tilde u = \min\big(\max(u,\ u_{\min}),\ u_{\max}\big), \qquad
g = -\log(-\log \tilde u),
$$

where $u_{\min} = 1.1754944 \times 10^{-38}$ is the smallest positive normal float32 and $u_{\max} = 0.99999994 = 1 - 2^{-24}$ is the largest float32 below `1.0`. Select the winning token index for each row `b` by taking the argmax over noisy logits:

$$
\text{token}_b = \operatorname*{argmax}_v \big( \text{logits}[b, v] + g[b, v] \big).
$$

The function returns a 1D integer tensor of length `B` with token indices in $[0, V)$.

Several properties follow from this definition and are checked by the tests:

- **A masked token is never returned.** Clamping bounds the noise to $-4.47 \le g \le 16.64$. A masked logit (`-inf`) therefore always scores `-inf`, even when `u = 1.0`, while a finite logit always has a finite score, even when `u = 0.0`.
- **Scores are never NaN.** Without the upper clamp, `u = 1.0` produces $g = +\infty$, and adding $+\infty$ to a masked `-inf` logit produces NaN.
- **Exact categorical distribution.** When `uniforms` contains independent samples from $[0, 1)$, token $v$ wins with probability $\operatorname{softmax}(\text{logits}[b])_v$.

Do not draw random numbers inside this function, and do not call your framework's categorical, multinomial, or Gumbel sampler.

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

In row 0, token 1 has the largest logit, but token 0 draws higher Gumbel noise and wins. Token 3 draws the largest noise of all, yet remains `-inf` because it is masked.

In row 1, all logits are equal, so the token with the largest uniform value wins.

In row 2, token 1 is the only unmasked entry and has `u = 0.0`. Clamping to $u_{\min}$ yields the smallest possible noise, $-\log(87.34) = -4.47$, keeping its score finite. Token 2 has `u = 1.0`, which clamps to $u_{\max}$ and yields the largest possible noise, `16.64`, so its score stays `-inf`. Without clamping, the scores would be `[-inf, -inf, nan, -inf]`.

## Constraints

- $1 \le B \le 20000$, $1 \le V \le 1000$, $0 \le u \le 1$. Finite logits lie in $[-1000, 1000]$.
- Output tokens must match the reference exactly. In every test row, the top score exceeds the second-best score by at least $10^{-3}$ (and by at least `1.0` when either token has `u = 0.0` or `u = 1.0`), so float32 rounding cannot flip a result.
- **Statistical cases.** The function is called once on 20,000 identical rows of logits with independent uniforms, and the empirical token frequencies are checked against the row's softmax probabilities using a finite-sample Chernoff bound. A correct solution fails with probability below $10^{-9}$, and fixed seeds make the check deterministic.
- No Python loop over rows or tokens is needed.

## Why this matters

Both `jax.random.categorical` and `mx.random.categorical` use this exact algorithm, as do most samplers that run on GPUs or inside compiled computation graphs. Taking an argmax over independent per-entry noise parallelizes cleanly without sorting, sequential cumulative sums, or softmax normalization, and it handles masked `-inf` entries directly. The same trick underpins Gumbel-softmax, Gumbel top-k sampling without replacement, and several LLM watermarking schemes.

The numerical boundaries are where production implementations break. A single decoding step with batch size 128 and a 150,000-token vocabulary consumes over $2^{24}$ uniform draws, so a float32 generator is expected to emit an exact `0.0` roughly once per step. Similarly, a single unclamped `1.0` on a masked token creates a NaN score that can cause `argmax` to return a token forbidden by a grammar.
