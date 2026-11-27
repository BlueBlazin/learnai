Work out the shapes on a single token first, then slice the projected tensor `raw` across all tokens into `raw[:, :N]`, `raw[:, N:2*N]`, and `raw[:, 2*N:]`, along with the matching slices of `base`. Reshape the last slice to `(T, N, N)` to form the matrix of logits.

---

Normalize the flattened vector of length `N * D` as a whole, using one mean square per token and no gain: `flat / sqrt(mean(flat**2, -1, keepdims) + 1e-6)`. Normalizing each stream separately erases the relative magnitudes across streams, which the projection needs to see. Keep `1e-6` fixed in the norm. Use `eps` only in its four places: added to `pre`, added to the softmax output, and added inside every row and column divisor.

---

Dividing every column by its sum means summing over axis `-2` (down the rows) with `keepdim=True`. Dividing every row by its sum means summing over axis `-1`. Run the initial column step first, then loop `iters - 1` times doing a row step followed by a column step. Verify on the worked example that after one iteration, the columns of `comb` sum to 1 while the rows do not.

---

For `hc_merge`, write out stream `n` of token `t`: `post[t, n] * y[t]` plus `sum_j comb[t, j, n] * streams[t, j]`. Because the sum contracts over the first matrix axis `j` of `comb`, the matrix product uses `comb` transposed: `comb.transpose(-1, -2) @ streams` (or `einsum("tjn,tjd->tnd", comb, streams)`). Broadcast the first term as `post[:, :, None] * y[:, None, :]`.
