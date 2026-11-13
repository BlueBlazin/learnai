Implement one verification step of speculative sampling: given the target model's
distribution `p`, the draft model's distribution `q`, and a token `x` that the draft
model sampled from `q`, decide whether to keep `x`, and if not, sample a replacement
so that the emitted token is distributed **exactly** as `p`. The rule is short, and
every part of it (the ratio, the residual, its normalization, the two independent
random numbers) is needed for that guarantee.

## Specification

```python
def speculative_step(p, q, x, u_accept, u_resample): ...
```

Randomness is an input: the function is deterministic, and the two uniform numbers
are supplied by the caller.

**1. Accept or reject.** Accept the draft token with probability
$\min\!\big(1,\; p[x] / q[x]\big)$, using `u_accept`:

$$
\text{accept} \iff u_\text{accept} \cdot q[x] < p[x]
\quad\Big(\text{equivalently } u_\text{accept} < \tfrac{p[x]}{q[x]}\Big).
$$

On acceptance return `(x, True)`. `u_resample` is not used.

**2. Resample on rejection.** Form the residual $r_i = \max(0,\; p_i - q_i)$, its
cumulative sums $c_k = \sum_{i \le k} r_i$, and its total $R = c_{V-1}$. Sample from
the normalized residual $r / R$ by inverse CDF, using `u_resample`:

$$
\text{token} = \min \{\, k : c_k > u_\text{resample} \cdot R \,\}.
$$

Return `(token, False)`. `u_accept` is not used again.

- `p`: float32 tensor of shape `(V,)`, the target model's next-token distribution
  (non-negative, sums to 1). Entries may be exactly 0.
- `q`: float32 tensor of shape `(V,)`, the draft model's distribution, with
  $q[x] > 0$ (the token was sampled from it).
- `x`: a Python int, the draft token.
- `u_accept`, `u_resample`: Python floats in $[0, 1)$.
- Returns `(token, accepted)`: the emitted token as a Python `int` and whether the
  draft token was accepted as a Python `bool`.

Things that follow from the definition, and that the tests check:

- If $p[x] \ge q[x]$ the token is accepted for every `u_accept`. If $p[x] = 0$ it is
  rejected for every `u_accept`, including `0.0`.
- A rejection implies $p[x] < q[x]$, hence $R > 0$: the residual can always be
  normalized. A resampled token always has $r_\text{token} > 0$, so it is never `x`
  and never a token with $p_i \le q_i$. In particular `u_resample = 0.0` gives the
  first token whose residual is positive.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(V,)` | float32 |
| `q` | `(V,)` | float32 |
| result | `(token, accepted)` | Python `int`, Python `bool` |

## Example

```python
p = [0.1, 0.4, 0.2, 0.3]     # target
q = [0.5, 0.2, 0.2, 0.1]     # draft
# residual max(0, p - q) = [0.0, 0.2, 0.0, 0.2],  R = 0.4,  c = [0.0, 0.2, 0.2, 0.4]

speculative_step(p, q, 0, 0.1, 0.9)    # (0, True)
speculative_step(p, q, 0, 0.6, 0.3)    # (1, False)
speculative_step(p, q, 0, 0.6, 0.7)    # (3, False)
speculative_step(p, q, 1, 0.99, 0.5)   # (1, True)
```

The draft model proposes token 0 half of the time, but the target gives it only 0.1,
so it is accepted with probability $0.1 / 0.5 = 0.2$. In the first call
$0.1 < 0.2$: accepted. In the second, $0.6 \ge 0.2$: rejected, and
$u_\text{resample} \cdot R = 0.3 \times 0.4 = 0.12$; the first cumulative sum above
0.12 is $c_1 = 0.2$, so token 1 is emitted. In the third, $0.7 \times 0.4 = 0.28$,
and the first cumulative sum above it is $c_3 = 0.4$: token 3. Token 2 has residual
0 and can never be the result of resampling. In the last call $p[1] \ge q[1]$, so
token 1 is accepted even with `u_accept = 0.99`.

Check the guarantee for token 1: it is emitted when it is drafted and accepted
(probability $0.2 \times 1$), or when another draft is rejected and resampling picks
it. Only token 0 can be rejected, with probability $0.5 \times 0.8 = 0.4$, and then
token 1 is chosen with probability $0.2 / 0.4$. In total
$0.2 + 0.4 \times 0.5 = 0.4 = p[1]$.

## Constraints

- $2 \le V \le 64$.
- **Exact cases.** Single calls are compared with the reference exactly (token and
  flag). Their inputs stay away from both decision boundaries, so float32 rounding
  cannot change the result: $|u_\text{accept} - p[x]/q[x]| \ge 10^{-3}$ whenever
  $0 < p[x]/q[x] < 1$, and $|u_\text{resample} - c_k / R| \ge 10^{-3}$ for every $k$
  with $c_k > 0$.
- **Statistical cases.** The test draws `x` from `q` and independent uniforms for up
  to 2000 calls and compares the empirical distribution of the emitted tokens with
  `p`, and the acceptance rate with $\sum_i \min(p_i, q_i)$, using finite-sample
  bounds (Chernoff for the distribution, Hoeffding for the rate). A correct solution
  fails with probability below $10^{-9}$, and since the seeds are fixed, it either
  always passes or always fails.
- The function is called a few thousand times per submit. It needs a handful of
  tensor operations per call and no Python loop over the vocabulary.
- Do not draw random numbers yourself.

## Why this matters

Speculative decoding lets a small draft model propose several tokens that the large
target model then verifies in a single parallel forward pass, which is typically two
to three times faster than decoding with the target alone. The selling point is that
it is lossless: the output distribution is that of the target model, not an
approximation. That property lives entirely in this function. Accept too eagerly, or
resample from `p` instead of the residual, and the system still produces fluent text
at the same speed, but from a distribution that is a blend of the two models: a
silent quality regression that no latency dashboard will show.
