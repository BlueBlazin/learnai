Implement one position of multi-draft speculative sampling. A draft model has proposed `n` candidate tokens for the same position (the children of one node in a token tree), and you must choose which candidate, if any, becomes the next token so that the output follows the target model's distribution `p` **exactly**. Testing every candidate with the single-draft rule breaks exactness. Instead, use *recursive rejection sampling*, where each rejection updates the distribution that the next candidate is tested against.

![This problem is Verify, for several candidate tokens at one position.](assets/map.svg)

## Specification

```python
def multi_draft_step(p, q, candidates, u_accept, u_resample): ...  # -> (token, n_rejected)
```

This is the multi-step speculative sampling algorithm from SpecInfer. Let $V$ be the vocabulary size and $n \ge 1$ be the number of candidates. The candidates $x_0, \dots, x_{n-1}$ are drawn **independently** from the same draft distribution `q` (sampling with replacement, so the same token may appear more than once). All randomness is passed in through `u_accept` and `u_resample`, making the function deterministic.

Maintain a *residual* distribution $r$ over the $V$ tokens, representing the probability mass that still needs to be delivered. Initialize $r^{(0)} = p$. Process the candidates in order. For candidate $i$ with token $x = x_i$:

1. **Test** candidate $i$ against the current residual using its dedicated uniform draw `u_accept[i]`:

$$
\text{candidate } i \text{ is accepted} \iff u_\text{accept}[i] \cdot q[x] < r^{(i)}[x].
$$

   This accepts with probability $\min\!\big(1,\ r^{(i)}[x] / q[x]\big)$. If candidate $i$ is accepted, immediately return `(x, i)`. Later candidates and their uniforms are ignored.

2. **Otherwise, update the residual** by subtracting the draft distribution, clipping negative entries to zero, and renormalizing:

$$
r^{(i+1)} = \frac{\max\big(0,\ r^{(i)} - q\big)}{\sum_{v=0}^{V-1} \max\big(0,\ r^{(i)}_v - q_v\big)}.
$$

If all `n` candidates are rejected, sample the final token from $r^{(n)}$ by inverse CDF using `u_resample`. With cumulative sums $c_j = \sum_{v \le j} r^{(n)}_v$ and total $W = c_{V-1}$, return `(token, n)` where

$$
\texttt{token} = \min \{\, j : c_j > u_\text{resample} \cdot W \,\}.
$$

**Input and output details:**

- `p` is the target distribution (non-negative, sums to 1; entries may be 0).
- `q` is the draft distribution, with $q[x_i] > 0$ for every candidate $x_i$.
- `u_accept[i]` lies in $[0, 1)$ and is used only for candidate `i`.
- `u_resample` lies in $[0, 1)$ and is used only when all `n` candidates are rejected.
- The returned `n_rejected` is the number of candidates rejected before emitting `token`. When `n_rejected < n`, candidate `n_rejected` was accepted and `token` is its token. When `n_rejected == n`, `token` was sampled from the final residual $r^{(n)}$.

Two properties follow directly from the definition, and the tests rely on them. First, rejecting $x$ implies $r^{(i)}[x] < q[x]$, so the denominator $\sum_v \max(0, r^{(i)}_v - q_v)$ is strictly positive and the residual can always be normalized. Second, rejecting $x$ sets $r^{(i+1)}[x] = 0$, so any later duplicate of $x$ is rejected even if its uniform is `0`, and $x$ can never be drawn as the final sample. A repeated candidate still counts as a stage and updates the residual again.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(V,)` | float32 |
| `q` | `(V,)` | float32 |
| `candidates` | `(n,)` | integer |
| `u_accept` | `(n,)` | float32 |
| `u_resample` | scalar | Python `float` |
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

- **First call:** Candidate 0 is token 0 with ratio $r^{(0)}[0] / q[0] = 0.1 / 0.5 = 0.2$. Since $0.1 < 0.2$, it is accepted immediately, returning `(0, 0)`.
- **Second call:** Since $0.6 \ge 0.2$, candidate 0 is rejected and the residual becomes $r^{(1)} = [0, 0.5, 0, 0.5]$. Candidate 1 is token 1 with ratio $r^{(1)}[1] / q[1] = 2.5$, so it is accepted for any uniform. (Here the ratio against the original `p` would have been 2, which would also accept, but for token 3 the ratio rises from 3 to 5, and for token 2 it drops from 1 to 0.)
- **Third call:** Candidate 0 is rejected. Candidate 1 is token 0 again, whose residual is now 0, so it is rejected even with `u_accept[1] == 0.0`. Two rejections leave $r^{(2)} = [0, 3/7, 0, 4/7]$ with cumulative sums $[0, 0.429, 0.429, 1]$. The first cumulative sum above `0.5` belongs to token 3, returning `(3, 2)`.
- **Fourth call:** Token 2 has $p[2] = q[2]$, so as the first candidate it is always accepted (as in the fifth call). After token 0 is rejected, however, token 2's residual drops to 0, so candidate 1 is rejected. Sampling from $r^{(2)}$ with `u_resample = 0.3 < 0.429` yields token 1, returning `(1, 2)`.

## Constraints

- $1 \le n \le 8$ and $2 \le V \le 64$.
- **Exact cases:** Single calls are compared for exact equality. Test inputs stay clear of every decision boundary so float32 rounding cannot flip the outcome. Every tested ratio $r^{(i)}[x]/q[x]$ in $(0, 1)$ differs from its uniform by at least $10^{-3}$, every positive residual entry is at least $10^{-6}$, and $|u_\text{resample} - c_j / W| \ge 10^{-3}$ for every $j$ with $c_j > 0$.
- **Statistical cases:** The harness draws `n` candidates independently from `q`, draws independent uniforms, and runs up to 700 calls per case. Using finite-sample Chernoff bounds, it checks that the emitted token follows `p`, that `n_rejected` follows the theoretical distribution, and that the fallback token after `n` rejections follows $r^{(n)}$. A correct solution fails with probability below $10^{-9}$, and because seeds are fixed, it either always passes or always fails.
- The function is called a few thousand times per submission, so a short loop over the candidates with a few tensor operations per step is expected.
- Do not draw random numbers yourself.

## Why this matters

A single chain of draft tokens stops at its first rejection. Tree-based speculative decoding hedges by proposing multiple candidates at each position, scoring the entire tree in one forward pass with a tree attention mask, and walking down from the root. Under greedy decoding, each step is a simple token comparison. Under sampling, each step along the accepted path runs this function, whose exactness guarantees that enabling tree speculation does not alter the model's output distribution.

The tempting shortcut, running the single-draft test against `p` for every child, accepts more often but corrupts the distribution. It shifts the output toward tokens the draft model favors, a subtle bias that standard fluency checks will not catch.
