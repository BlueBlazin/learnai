One candidate is the ordinary speculative sampling step: accept $x$ with probability
$\min(1, p[x]/q[x])$, otherwise the token must come from the normalized residual
$\max(0, p - q)$. In that step "sample from the residual" means drawing from it
directly. With a second candidate in hand, could it serve as an attempt to sample
from the residual? What would it have to be tested against?

---

After a rejection the job is the same job with a new target: produce a sample from
$r' = \operatorname{norm}(\max(0, r - q))$, given a fresh draw from $q$. So the
rule is recursive. Keep one vector `residual`, starting at `p`. For each candidate:
accept if `u_accept[i] * q[x] < residual[x]`; otherwise replace `residual` by
`max(0, residual - q)` divided by its sum. The test always uses the *current*
residual and the *original* `q` (the candidates are independent draws from it).

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

Return as soon as a candidate is accepted, and use `u_resample` only at the end.
Normalizing matters: the acceptance test compares against a probability, and an
unnormalized residual accepts too rarely.
