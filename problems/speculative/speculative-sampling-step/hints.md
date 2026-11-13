Start from the goal: the emitted token must have distribution `p`, although the
proposal came from `q`. A token is emitted either because it was drafted and
accepted, or because some draft was rejected and resampling chose it. If drafts are
accepted with probability $\min(1, p[x]/q[x])$, how much probability does token $t$
receive through the first route, and how much is still missing compared with $p[t]$?

---

Through acceptance, token $t$ receives $q[t] \cdot \min(1, p[t]/q[t]) = \min(p[t],
q[t])$. What is missing is $p[t] - \min(p[t], q[t]) = \max(0, p[t] - q[t])$. So the
resampling distribution must be proportional to $\max(0, p - q)$: zero wherever the
draft model already supplies enough, positive where it undersupplies. It has to be
normalized before you can sample from it, and its total is less than 1.

---

Two decisions, two uniforms. Acceptance: `u_accept * q[x] < p[x]` (no division, no
`min`). On rejection: `c = cumsum(clamp(p - q, min=0))`, and the token is the first
index with `c[k] > u_resample * c[-1]`. Scaling the uniform by the total `c[-1]`
is the normalization. The first index that exceeds a threshold equals the number of
entries that do not: `sum(c <= threshold)`. Convert to a Python `int` and return the
pair.
