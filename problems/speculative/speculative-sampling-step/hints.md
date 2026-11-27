Start from the goal. The emitted token must follow `p`, even though the proposal came
from `q`. A token can be emitted either because it was drafted and accepted, or
because some draft was rejected and resampling picked it. If draft token `x` is
accepted with probability $\min(1, p[x] / q[x])$, how much probability does token $t$
get from acceptance, and how much is still missing compared with $p[t]$?

---

Through acceptance, token $t$ receives
$q[t] \cdot \min(1, p[t] / q[t]) = \min(p[t], q[t])$. The missing probability is
$p[t] - \min(p[t], q[t]) = \max(0, p[t] - q[t])$. The resampling distribution must
therefore be proportional to $\max(0, p - q)$, which is zero wherever the draft model
already supplies enough mass and positive where it undersupplies. Remember that its
sum is less than 1, so it must be normalized before sampling.

---

Use two separate uniforms for the two decisions. Test acceptance with
`u_accept * q[x] < p[x]`, which avoids both division and `min`. On rejection, compute
`c = cumsum(clamp(p - q, min=0))` and find the first index where
`c[k] > u_resample * c[-1]`. Multiplying `u_resample` by `c[-1]` performs the
normalization, and the first index exceeding the threshold equals `sum(c <= threshold)`.
Convert the token to a Python `int` and return `(token, accepted)`.
