Each draft token has its own acceptance test, exactly the one from a single
speculative step, and you can evaluate all `K` tests at once. But "draft 2 passed
its test" is not the same as "draft 2 is accepted". Draft 2 was sampled by the draft
model *after* drafts 0 and 1. What is its test worth if draft 1 was rejected?

---

Gather the probabilities of the drafted tokens, `p[i, x_i]` and `q[i, x_i]` for
`i` in `0..K-1`, and form the boolean vector
`passes = u_accept * q_x < p_x`. `n_accepted` is the length of its leading run of
`True`: the index of the first `False`, or `K` if there is none. A cumulative
product of the 0/1 vector, summed, gives exactly that. Then exactly one token is
sampled, and the only question is from which weights.

---

If `n < K`, position `n` was rejected: sample from `max(0, p[n] - q[n])`, as in the
single step. If `n == K`, no draft exists for row `K`, so there is nothing to
subtract: sample from `p[K]` itself. Either way it is one inverse-CDF draw with
`u_resample`: `c = cumsum(weights)`, and the token is the number of entries with
`c <= u_resample * c[-1]`. (Appending a row of zeros to `q` makes the two cases the
same expression, `max(0, p[n] - q[n])`.) Return two Python ints.
