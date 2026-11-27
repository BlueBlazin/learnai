Each draft token uses the same acceptance test as a single speculative step, and you can evaluate all `K` tests at once. However, "draft 2 passed its test" does not mean "draft 2 is accepted." Draft 2 was sampled by the draft model *after* drafts 0 and 1, so ask yourself whether its test is still valid if draft 1 was rejected.

---

Gather the probabilities of the drafted tokens, `p[i, x_i]` and `q[i, x_i]` for `i` in `0..K-1`, and compute the boolean vector `passes = u_accept * q_x < p_x`. Then `n_accepted` is the length of the leading run of `True` values (the index of the first `False`, or `K` if all pass). Casting `passes` to integers, taking its cumulative product, and summing gives that count directly. After that, you sample one token, and the only choice is which weight vector to use.

---

If `n_accepted < K`, position `n_accepted` was rejected, so sample from `max(0, p[n_accepted] - q[n_accepted])` as in the single-step problem. If `n_accepted == K`, no draft exists at row `K`, so sample from `p[K]` directly. In both cases, run one inverse-CDF draw with `u_resample` by computing `c = cumsum(weights)` and counting how many entries satisfy `c <= u_resample * c[-1]`. (Appending a row of zeros to `q` unifies both branches into `max(0, p[n_accepted] - q[n_accepted])`.) Return `(n_accepted, next_token)` as two Python `int`s.
