Implement one position of multi-draft speculative sampling: the draft model proposed
`n` candidate tokens for the same position (the children of one node in a token
tree), and you decide which of them, if any, becomes the next token, such that the
emitted token is distributed **exactly** as the target model's distribution `p`.
Testing every candidate with the single-draft rule is not exact. The scheme that is
exact is *recursive rejection sampling*: every rejection changes the distribution
the next candidate is tested against.

## Specification

```python
def multi_draft_step(p, q, candidates, u_accept, u_resample): ...  # -> (token, n_rejected)
```

This is the multi-step speculative sampling of SpecInfer, for candidates
$x_0, \dots, x_{n-1}$ drawn **independently** from the same draft distribution `q`
(sampling with replacement: the same token may appear more than once). Randomness
is an input: the function is deterministic.

Keep a *residual* distribution $r$, the distribution that still has to be
delivered. It starts as the target: $r^{(0)} = p$. Go through the candidates in
order; for candidate $i$ with token $x = x_i$:

1. **Test** it against the current residual, with its own uniform:

$$
\text{candidate } i \text{ is accepted} \iff u_\text{accept}[i] \cdot q[x] < r^{(i)}[x],
$$

   that is, with probability $\min\!\big(1,\ r^{(i)}[x] / q[x]\big)$. If it is
   accepted, return `(x, i)`: later candidates and their uniforms have no effect.

2. **Otherwise update the residual**: subtract the draft distribution, clip at
   zero, and normalize.

$$
r^{(i+1)} = \frac{\max\big(0,\ r^{(i)} - q\big)}{\sum_v \max\big(0,\ r^{(i)}_v - q_v\big)}
$$

If all `n` candidates are rejected, sample the token from the last residual
$r^{(n)}$ by inverse CDF with `u_resample`: with cumulative sums
$c_j = \sum_{v \le j} r^{(n)}_v$ and total $W = c_{V-1}$,

$$
\texttt{token} = \min \{\, j : c_j > u_\text{resample} \cdot W \,\},
$$

and return `(token, n)`.

- `p`: float32 tensor of shape `(V,)`, the target's distribution (non-negative, sums
  to 1). Entries may be exactly 0.
- `q`: float32 tensor of shape `(V,)`, the draft's distribution, with
  $q[x_i] > 0$ for every candidate.
- `candidates`: integer tensor of shape `(n,)`, $n \ge 1$.
- `u_accept`: float32 tensor of shape `(n,)`, values in $[0, 1)$; `u_accept[i]`
  is used for candidate `i` only.
- `u_resample`: a Python float in $[0, 1)$, used only when every candidate is
  rejected.
- Returns `(token, n_rejected)` as two Python `int`s: the emitted token, and the
  number of candidates rejected before it. `n_rejected < n` means candidate
  `n_rejected` was accepted (and `token` is its token); `n_rejected == n` means the
  token was sampled from the last residual.

Two facts follow from the definition, and the tests rely on them. A rejection of
$x$ implies $r^{(i)}[x] < q[x]$, so the sum in the denominator is positive: the
residual can always be normalized. And it implies $r^{(i+1)}[x] = 0$: a token that
was rejected once is rejected again wherever it reappears among the candidates
(even with a uniform of 0), and is never the final sample. A repeated candidate
still counts as a stage: the residual is updated again.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(V,)` | float32 |
| `q` | `(V,)` | float32 |
| `candidates` | `(n,)` | integer |
| `u_accept` | `(n,)` | float32 |
| result | `(token, n_rejected)` | Python `int`, Python `int` |

## Example

```python
p = [0.1, 0.4, 0.2, 0.3]  # target
q = [0.5, 0.2, 0.2, 0.1]  # draft
# r0 = p
# r1 = norm(max(0, r0 - q)) = norm([0, 0.2, 0, 0.2]) = [0, 0.5, 0, 0.5]
# r2 = norm(max(0, r1 - q)) = norm([0, 0.3, 0, 0.4]) = [0, 3/7, 0, 4/7]

multi_draft_step(p, q, [0, 1], [0.1, 0.9], 0.5)  # (0, 0)
multi_draft_step(p, q, [0, 1], [0.6, 0.9], 0.5)  # (1, 1)
multi_draft_step(p, q, [0, 0], [0.6, 0.0], 0.5)  # (3, 2)
multi_draft_step(p, q, [0, 2], [0.6, 0.3], 0.3)  # (1, 2)
multi_draft_step(p, q, [2, 0], [0.99, 0.1], 0.5)  # (2, 0)
```

First call. Candidate 0 is token 0 with $r^{(0)}[0] / q[0] = 0.1 / 0.5 = 0.2$, and
$0.1 < 0.2$: accepted at once.

Second call. $0.6 \ge 0.2$: candidate 0 is rejected and the residual becomes
$r^{(1)} = [0, 0.5, 0, 0.5]$. Candidate 1 is token 1 with
$r^{(1)}[1] / q[1] = 2.5$: accepted for any uniform. Against the original `p` its
ratio would have been 2 here, which makes no difference; for token 3 the ratio
rises from 3 to 5, and for token 2 it drops from 1 to 0.

Third call. Candidate 0 is rejected. Candidate 1 is token 0 again, whose residual
is now exactly 0: rejected although its uniform is 0. Two rejections leave
$r^{(2)} = [0, 3/7, 0, 4/7]$ with cumulative sums $[0, 0.429, 0.429, 1]$; the
first one above 0.5 belongs to token 3.

Fourth call. Token 2 has $p = q$, so as a first candidate it would always be
accepted (fifth call). As a second candidate, after token 0 was rejected, its
residual is 0: rejected. The final sample with $u = 0.3 < 0.429$ is token 1.

## Constraints

- $1 \le n \le 8$ and $2 \le V \le 64$.
- **Exact cases.** Single calls are compared exactly. Their inputs stay away from
  every decision boundary, so float32 rounding cannot change the result: every
  tested ratio $r^{(i)}[x]/q[x]$ in $(0, 1)$ differs from its uniform by at least
  $10^{-3}$, every positive residual entry is at least $10^{-6}$, and
  $|u_\text{resample} - c_j / W| \ge 10^{-3}$ for every $j$ with $c_j > 0$.
- **Statistical cases.** The test draws the `n` candidates independently from `q`,
  draws independent uniforms, and makes up to 700 calls per case. It compares the
  distribution of the emitted token with `p`, the distribution of `n_rejected`
  with the one the scheme implies, and the token after `n` rejections with
  $r^{(n)}$, using finite-sample Chernoff bounds. A correct solution fails with
  probability below $10^{-9}$, and since the seeds are fixed, it either always
  passes or always fails.
- The function is called a few thousand times per submit: a short loop over the
  candidates with a handful of tensor operations each is expected.
- Do not draw random numbers yourself.

## Why this matters

A single chain of draft tokens dies at its first rejection. Tree-based speculative
decoding hedges: at each position the draft proposes several tokens, the target
scores the whole tree in one pass with a tree attention mask, and verification
walks down from the root, choosing at each node among the children. With greedy
decoding that choice is a comparison. With sampling it is this function, once per
node on the accepted path, and its exactness is what lets a server turn tree
speculation on without changing what the model says. The tempting shortcut, giving
every child the single-draft test against `p`, accepts more often and is wrong: the
output drifts toward tokens the draft likes, by an amount no benchmark of fluency
will notice.
