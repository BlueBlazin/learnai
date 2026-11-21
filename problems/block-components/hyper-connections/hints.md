Work on one token first and name the three slices of the projected vector before doing
anything with them: `raw[:, :N]`, `raw[:, N:2*N]`, `raw[:, 2*N:]`, and the same three
slices of `base`. The last one becomes a matrix with `reshape(T, N, N)`.

---

The norm is over the flattened vector of length `N * D`, one mean square per token,
and it has no gain: `flat / sqrt(mean(flat**2, -1, keepdims) + 1e-6)`. Normalizing each
stream separately throws away the relative sizes of the streams, which is information
the projection is meant to see. The `1e-6` is a constant of the model; `eps` appears
in four other places: added to `pre`, added to the softmax, and inside every row and
column divisor.

---

"Divide every column by its sum" is a sum over axis `-2` (down the rows) with
`keepdim`, and "every row" a sum over axis `-1`. Write the first column step, then a
loop that runs `iters - 1` times with a row step and a column step. Check yourself on
the example: after one iteration the columns of `comb` sum to 1 and the rows do not.

---

For the merge, write out stream `n` of the result: `post[t, n] * y[t]` plus
`sum_j comb[t, j, n] * streams[t, j]`. The index that is summed is the *first* index
of `comb`, so as a matrix product it is `comb` transposed times the streams:
`comb.transpose(-1, -2) @ streams`, or `einsum("tjn,tjd->tnd", comb, streams)`. The
first term needs `post[:, :, None] * y[:, None, :]`.
