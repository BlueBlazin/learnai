## The idea

Start with the single-draft speculative sampling step. Given a target distribution $r$ and a draft proposal $x \sim q$, we accept $x$ with probability $\min(1, r[x] / q[x])$. The probability of emitting token $v$ by acceptance is

$$
q_v \cdot \min\!\big(1, r_v / q_v\big) = \min(r_v, q_v).
$$

Summing over all tokens gives the total acceptance probability $\beta = \sum_v \min(r_v, q_v)$. What acceptance fails to cover is the deficit $r_v - \min(r_v, q_v) = \max(0, r_v - q_v)$, which has total mass $1 - \beta$. In the single-draft step, we cover this deficit by sampling directly from the normalized residual:

$$
r' = \frac{\max(0, r - q)}{1 - \beta}.
$$

The key insight behind multi-draft sampling is that sampling from $r'$ is the exact same problem we started with, just with $r'$ in place of $r$. When another independent draft proposal from $q$ is available, we test it against $r'$ using the same rule. We only sample from the residual directly once all proposals are exhausted.

## Why it is exact

**Claim.** For every distribution $r$ and every $n \ge 0$, starting the procedure with residual $r$ and $n$ independent candidates from $q$ emits a token distributed as $r$.

**Proof** by induction on $n$. For $n = 0$, the token is sampled directly from $r$. For $n \ge 1$, condition on the first candidate $x \sim q$ and its uniform draw:

- Candidate 0 is accepted and emitted as token $v$ with probability $\min(r_v, q_v)$.
- Candidate 0 is rejected with probability $1 - \beta$. Because the remaining $n - 1$ candidates and all remaining uniforms are independent of this rejection, the procedure continues from residual $r'$ with $n - 1$ fresh candidates. By the induction hypothesis, it emits token $v$ with probability $r'_v$.

Adding both cases gives

$$
P(\text{token} = v) = \min(r_v, q_v) + (1 - \beta) \cdot \frac{\max(0, r_v - q_v)}{1 - \beta}
= \min(r_v, q_v) + \max(0, r_v - q_v) = r_v .
$$

Setting $r = p$ proves that the emitted token follows $p$ for any draft distribution $q$ and any number of candidates $n$. $\blacksquare$

Three parts of this proof are load-bearing, and breaking any of them introduces bias:

- After a rejection, the next stage must target $r'$, not $r$ and not $p$.
- The residual $r'$ must be normalized, because the acceptance test $u \cdot q[x] < r'[x]$ compares against a probability distribution.
- Each rejection must be independent of all later steps, so each candidate needs its own uniform draw and the final resample needs one more.

Notice that $q$ never changes. All candidates are independent draws from the same draft distribution, so every stage tests against the original `q`. This also explains duplicate candidates. If token $x$ is rejected, then $r[x] < q[x]$ and therefore $r'[x] = 0$. Any later copy of $x$ is rejected with certainty and still advances the residual, which is exactly what the proof requires since it is another independent draw from $q$.

The stage acceptance probabilities $\beta_i = \sum_v \min(r^{(i)}_v, q_v)$ form a fixed sequence, and the probability that all $n$ candidates are rejected is $\prod_i (1 - \beta_i)$. The first factor $1 - \beta_0$ is the total variation distance between $p$ and $q$, matching single-draft sampling, while extra candidates reduce rejection probability through the later factors.

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

The loop is inherently sequential because stage $i$ is only reached if stage $i - 1$ rejected. However, because the residuals $r^{(0)}, \dots, r^{(n)}$ do not depend on which candidates were drawn, a batched implementation can precompute them for every node and gather as needed. The JAX and MLX implementations are line-by-line translations.

## Common mistakes

- **Testing every candidate against `p`.** Running the single-draft test on each candidate and falling back to `p` delivers $\min(p, q)$ repeatedly instead of the remaining deficit. Any token overrated by the draft ($q_v > p_v$) collects its full mass $p_v$ at every stage reached, skewing the output toward the draft model.
- **Updating only the fallback.** Testing every candidate against `p` and sampling from $r^{(1)}$ at the end makes the same error with a different tail.
- **Leaving the residual unnormalized.** Because $\max(0, r - q)$ has total mass $1 - \beta$, using it directly in the next test accepts with probability $\min(1, (1 - \beta)\,r'[x] / q[x])$, which is too low and shifts mass to the wrong tokens. (An unnormalized residual only works if you scale the left-hand side by its sum: `u * q[x] * total < residual[x]`.)
- **Sampling the fallback from `p`.** After $n$ rejections, the fallback must come from $r^{(n)}$. Sampling from `p` puts probability mass back onto tokens that were already rejected.
- **Not stopping at the first acceptance.** Later candidates target residuals that only exist after a rejection, so returning the last passing candidate distorts the distribution.
- **Zeroing only the rejected token.** Setting `residual[x] = 0` and renormalizing confuses this update with updating the *draft* distribution under sampling without replacement (see below). The target residual must subtract all of `q`.
- **Reusing a uniform or drawing random numbers internally.** Each stage must use its own `u_accept[i]`, and the function must stay deterministic by using only the provided uniforms.

## Complexity

The worst-case time complexity is $O(n V)$ for one residual update per rejection, with $O(V)$ memory. This cost is negligible compared to the target model's forward pass over the token tree.

## In the wild

- This algorithm is the multi-step speculative sampling procedure from "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification" (ASPLOS 2024, arXiv:2305.09781), Algorithm 2. At each tree node, SpecInfer picks a remaining child at random, accepts it if $u \le P_\text{LLM}(x) / P_\text{SSM}(x)$ for a uniform draw $u$, and otherwise replaces $P_\text{LLM}$ with $\operatorname{norm}(\max(0, P_\text{LLM} - P_\text{SSM}))$ and removes that child. When no children remain, it samples from the adjusted distribution. Theorem 4.2 proves exact equivalence with the target LLM and shows that the rejection probability is no larger than under the naive rule. Because the independent children are exchangeable, testing them in their given order is identical.
- Drawing candidates **without replacement** avoids wasted duplicates. "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding" (arXiv:2402.12374) updates the residual the same way after rejecting token $x$, and also zeros out $x$ in the draft distribution and renormalizes it before drawing and testing the next child. The induction proof still holds with each stage's updated draft distribution in place of $q$. "Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement" (arXiv:2402.14160) builds these trees using the Gumbel-top-k trick and stochastic beam search.
- The verifier must always match how the drafter sampled. Testing without-replacement candidates against an un-updated `q`, or vice versa, silently shifts the output distribution.
