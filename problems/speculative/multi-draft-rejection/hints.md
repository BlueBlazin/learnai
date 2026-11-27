Start from the single-draft speculative sampling step. You accept a proposal $x$ with probability $\min(1, p[x] / q[x])$, and if it is rejected, you sample from the normalized residual $\max(0, p - q)$. When a second candidate from $q$ is already available, think about how you can use it to attempt that sample from the residual instead of drawing directly. What distribution must the second candidate be tested against?

---

After a rejection, you face the exact same problem with a new target distribution: produce a sample from $r' = \operatorname{norm}(\max(0, r - q))$ given a fresh draw from $q$. Maintain a single vector `residual`, initialized to `p`. For each candidate `x`, accept if `u_accept[i] * q[x] < residual[x]`. Otherwise, replace `residual` with `max(0, residual - q)` divided by its sum. Always test against the *current* residual and the *original* `q`, since every candidate is an independent draw from `q`.

---

```
residual = p
for i, x in enumerate(candidates):
    if u_accept[i] * q[x] < residual[x]:
        return x, i
    residual = max(0, residual - q)
    residual = residual / sum(residual)
cdf = cumsum(residual)
return count(cdf <= u_resample * cdf[-1]), n
```

Return immediately on the first acceptance, and use `u_resample` only if all candidates are rejected. Normalizing `residual` at every step is essential because the acceptance test compares against a valid probability distribution. An unnormalized residual accepts too rarely.
