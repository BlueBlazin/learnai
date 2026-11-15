## The idea

Recall the single step. A proposal $x \sim q$ is accepted with probability
$\min(1, r[x]/q[x])$, where $r$ is the distribution we want. The probability of
emitting token $v$ *by acceptance* is

$$
q_v \cdot \min\!\big(1, r_v / q_v\big) = \min(r_v, q_v),
$$

so acceptance happens with probability $\beta = \sum_v \min(r_v, q_v)$, and what
acceptance fails to deliver is $r_v - \min(r_v, q_v) = \max(0, r_v - q_v)$, of
total mass $1 - \beta$. In the single step that deficit is paid by sampling from
its normalized form,

$$
r' = \frac{\max(0, r - q)}{1 - \beta},
$$

directly. **The observation behind multi-draft sampling is that "produce a sample
from $r'$" is the problem we started with**, with $r'$ in place of $r$. If another
independent proposal from $q$ is available, use it to attack that problem, by the
same rule. Only when the proposals run out is the residual sampled directly.

## Why it is exact

**Claim.** For every distribution $r$ and every $n \ge 0$, the procedure started
with residual $r$ and $n$ independent candidates from $q$ emits a token distributed
as $r$.

**Proof** by induction on $n$. For $n = 0$ the token is sampled from $r$ itself.
For $n \ge 1$, split on the first candidate $x \sim q$ and its uniform:

- it is accepted and emitted with probability $\min(r_v, q_v)$ for $x = v$;
- it is rejected with probability $1 - \beta$. Its uniform and the other
  $n - 1$ candidates are independent of that event, so the procedure continues
  from residual $r'$ with $n - 1$ fresh candidates, and by the induction hypothesis
  emits $v$ with probability $r'_v$.

Together,

$$
P(\text{token} = v) = \min(r_v, q_v) + (1 - \beta) \cdot \frac{\max(0, r_v - q_v)}{1 - \beta}
= \min(r_v, q_v) + \max(0, r_v - q_v) = r_v .
$$

Starting from $r = p$ gives the theorem: the output follows $p$ for any $q$ and
any number of candidates. $\blacksquare$

Three things in the proof are load-bearing, and each corresponds to a way of
getting it wrong:

- the continuation targets $r'$, not $r$ and not $p$ (the residual must be
  **updated**);
- $r'$ is a distribution (the residual must be **normalized**, because the test
  $u \cdot q[x] < r'[x]$ compares against a probability);
- the rejection event must be independent of everything used afterwards (each
  candidate has its **own uniform**, and the final sample has one more).

Note what does *not* change: $q$. The candidates are independent draws from the
same distribution, so every stage tests against the same `q`. That also explains
the behavior on duplicates. If $x$ is rejected then $r[x] < q[x]$, so
$r'[x] = 0$: a second copy of $x$ is rejected with certainty, and it still
advances the residual, exactly as the proof requires (it is one more independent
proposal, and it happened to be useless).

The acceptance probabilities $\beta_0, \beta_1, \dots$ with
$\beta_i = \sum_v \min(r^{(i)}_v, q_v)$ form a fixed sequence, and the chance that
all $n$ candidates fail is $\prod_i (1 - \beta_i)$. The first factor is the total
variation distance between $p$ and $q$: the same as for a single draft. More
candidates only help through the later factors.

## Reference solution (PyTorch)

```python
def multi_draft_step(p, q, candidates, u_accept, u_resample):
    residual = p
    n = candidates.shape[0]
    for i in range(n):
        x = candidates[i]
        if bool(u_accept[i] * q[x] < residual[x]):
            return int(x), i
        residual = torch.clamp(residual - q, min=0.0)
        residual = residual / residual.sum()
    cdf = torch.cumsum(residual, dim=0)
    return int((cdf <= u_resample * cdf[-1]).sum()), n
```

The loop is sequential by nature: stage $i$ exists only if stage $i - 1$ rejected.
(The residuals themselves do not depend on the candidates, so a batched
implementation can precompute $r^{(0)}, \dots, r^{(n)}$ for every node and gather.)
The JAX and MLX versions are line-by-line translations.

## Common mistakes

- **Independent tests against `p`.** Give every candidate the single-draft test
  and fall back to sampling from `p`. Every stage then delivers $\min(p, q)$
  again instead of what is still missing, so a token that the draft overrates
  ($q_v > p_v$) collects its full probability $p_v$ at each stage that is
  reached: the output leans toward the draft model.
- **Updating nothing but the fallback.** Testing every candidate against `p` and
  sampling the first residual at the end is the same error with a different tail.
- **Carrying the residual unnormalized.** `max(0, r - q)` has mass $1 - \beta$.
  Used as it is in the next test, it accepts with probability
  $\min(1, (1-\beta)\,r'[x]/q[x])$, too rarely, and the deficit ends up in the
  wrong tokens. (An unnormalized residual is fine if the test carries the total:
  `u * q[x] * total < residual[x]`.)
- **Sampling the fallback from `p`.** After $n$ rejections the token must come
  from $r^{(n)}$; `p` puts mass back on tokens that were just rejected.
- **Not stopping at the first acceptance.** Later candidates were meant for a
  residual that only exists after a rejection. Returning the last candidate that
  passes changes the distribution.
- **Removing only the rejected token.** Setting `residual[x] = 0` and
  renormalizing confuses this scheme with the update of the *draft* distribution
  in sampling without replacement (below). The target side always loses all of
  `q`.
- **Reusing a uniform**, or drawing random numbers inside the function.

## Complexity

$O(n V)$ time in the worst case (one residual update per rejection) and $O(V)$
memory; negligible next to the target's forward pass over the token tree.

## In the wild

- The scheme is the multi-step speculative sampling of "SpecInfer: Accelerating
  Large Language Model Serving with Tree-based Speculative Inference and
  Verification" (ASPLOS 2024, arXiv:2305.09781), Algorithm 2: at each node of the
  token tree, pick one of the remaining children at random, accept it if
  $r \le P_\text{LLM}(x) / P_\text{SSM}(x)$, otherwise replace the LLM's
  distribution by $\operatorname{norm}(\max(0, P_\text{LLM} - P_\text{SSM}))$ and
  remove that child; when no child is left, sample from the adjusted
  distribution. The paper states the equivalence with the LLM's own sampling as
  its Theorem 4.2 and shows that the rejection probability is no larger than that
  of the naive rule. Since the children are exchangeable, taking them in the
  given order is the same procedure.
- Drawing the candidates **without replacement** removes the wasted duplicates.
  "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding"
  (arXiv:2402.12374) does this: after rejecting token $x$ it updates the residual
  as here, and additionally sets the draft's probability of $x$ to zero and
  renormalizes the draft distribution, from which the next child is drawn and
  against which it is tested. The induction above goes through unchanged with the
  stage's own draft distribution in place of $q$. "Recursive Speculative Decoding:
  Accelerating LLM Inference via Sampling Without Replacement" (arXiv:2402.14160)
  builds such trees with the Gumbel-top-k trick and stochastic beam search.
- Which scheme a verifier implements must match how the drafter sampled: testing
  without-replacement candidates against an un-updated `q`, or the reverse, is
  again a silent distribution shift.
