Implement one verification step of speculative sampling. Given the target model's
next-token distribution `p`, the draft model's distribution `q`, and a draft token
`x` sampled from `q`, decide whether to accept `x` or sample a replacement token so
that the emitted token follows `p` **exactly**.

![This problem is Verify, which keeps or replaces one sampled draft token.](assets/map.svg)

## Specification

```python
def speculative_step(p, q, x, u_accept, u_resample): ...
```

The function is deterministic. Instead of drawing random numbers inside the function,
the caller passes two uniform random floats in $[0, 1)$, `u_accept` and `u_resample`.

Let $V$ be the vocabulary size. The inputs are:

- `p`: the target model's next-token probability distribution over the $V$ tokens.
  Entries are non-negative and sum to 1, and some entries may be `0.0`.
- `q`: the draft model's next-token probability distribution over the same $V$
  tokens, with $q[x] > 0$ because `x` was sampled from `q`.
- `x`: the drafted token index.
- `u_accept`, `u_resample`: two independent uniform numbers in $[0, 1)$.

The function returns `(token, accepted)`, where `token` is the emitted token index
and `accepted` is a boolean indicating whether the draft token was kept.

**1. Accept or reject.** Accept the draft token `x` with probability
$\min(1, p[x] / q[x])$. Using `u_accept`, accept if and only if

$$
u_\text{accept} \cdot q[x] < p[x]
\quad\Big(\text{equivalently } u_\text{accept} < \frac{p[x]}{q[x]}\Big).
$$

When accepted, return `(x, True)` and ignore `u_resample`.

**2. Resample on rejection.** Compute the residual distribution
$r_i = \max(0, p_i - q_i)$ for each token $i \in \{0, \dots, V - 1\}$. Form its
cumulative sums $c_k = \sum_{i \le k} r_i$ and total mass $R = c_{V-1}$. Sample from
the normalized residual $r / R$ by inverse CDF using `u_resample`:

$$
\text{token} = \min \{\, k : c_k > u_\text{resample} \cdot R \,\}.
$$

Return `(token, False)` and do not reuse `u_accept`.

Two consequences follow directly from these rules and are checked by the tests:

- If $p[x] \ge q[x]$, the draft token is accepted for every `u_accept` in $[0, 1)$.
  If $p[x] = 0$, it is rejected for every `u_accept`, including `0.0`.
- Rejection only occurs when $p[x] < q[x]$, which guarantees $R > 0$, so the residual
  can always be normalized. Any resampled token has $r_\text{token} > 0$, so it is
  never `x` and never a token with $p_i \le q_i$. In particular, `u_resample = 0.0`
  selects the first token with a positive residual.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(V,)` | float32 |
| `q` | `(V,)` | float32 |
| `x` | scalar | Python `int` |
| `u_accept`, `u_resample` | scalar | Python `float` |
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

The draft model proposes token 0 with probability 0.5, while the target assigns it
only 0.1, so token 0 is accepted with probability $0.1 / 0.5 = 0.2$.

- In the first call, $0.1 < 0.2$, so token 0 is accepted.
- In the second call, $0.6 \ge 0.2$, so token 0 is rejected. The threshold is
  $u_\text{resample} \cdot R = 0.3 \times 0.4 = 0.12$. The first cumulative sum above
  0.12 is $c_1 = 0.2$, so token 1 is emitted.
- In the third call, the threshold is $0.7 \times 0.4 = 0.28$. The first cumulative
  sum above 0.28 is $c_3 = 0.4$, so token 3 is emitted. Token 2 has residual 0 and
  can never be chosen by resampling.
- In the fourth call, $p[1] \ge q[1]$, so token 1 is accepted even when
  `u_accept = 0.99`.

You can verify the exactness guarantee on token 1. It is emitted either when drafted
and accepted (probability $0.2 \times 1 = 0.2$) or when another draft is rejected and
resampling picks it. Only token 0 can be rejected, which happens with probability
$0.5 \times 0.8 = 0.4$, and resampling then picks token 1 with probability
$0.2 / 0.4 = 0.5$. The total probability is $0.2 + 0.4 \times 0.5 = 0.4 = p[1]$.

![The target p and a draft q over eight tokens. The filled part of a bar is accepted; it and the residual above it stack to p, whatever the draft.](interactive:draft-overlap)

## Constraints

- $2 \le V \le 64$.
- **Exact cases.** Single calls are compared against the reference for exact equality
  on both `token` and `accepted`. Test inputs stay away from decision boundaries so
  float32 rounding cannot flip the outcome. Specifically,
  $|u_\text{accept} - p[x]/q[x]| \ge 10^{-3}$ whenever $0 < p[x]/q[x] < 1$, and
  $|u_\text{resample} - c_k / R| \ge 10^{-3}$ for every $k$ with $c_k > 0$.
- **Statistical cases.** The test draws `x` from `q` and independent uniforms for up
  to 2000 calls. It checks the empirical token distribution against `p` with a
  Chernoff bound and checks the acceptance rate against $\sum_i \min(p_i, q_i)$ with a
  Hoeffding bound. A correct solution fails with probability below $10^{-9}$, and
  because the seeds are fixed, it either always passes or always fails.
- The function is called a few thousand times per submission. Use a handful of tensor
  operations per call and avoid a Python loop over the vocabulary.
- Do not call any random number generator inside `speculative_step`.

## Why this matters

Speculative decoding uses a small draft model to propose several tokens and a large
target model to verify them in one parallel forward pass, typically speeding up
decoding by 2x to 3x. Its main guarantee is that decoding remains lossless, meaning
the output distribution matches the target model exactly rather than approximating
it.

That guarantee rests entirely on this verification step. If you accept too eagerly or
resample from `p` instead of the residual, the decoder still produces fluent text at
high speed, but from a blended distribution of the two models. That creates a silent
quality regression that latency dashboards will never catch.
