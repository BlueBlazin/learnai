Start from the single-sequence acceptance algorithm and check what shape each
intermediate tensor takes once a batch axis of size $B$ is added in front. The
acceptance tests become a `(B, K)` boolean matrix. Finding the first failure is now
a per-row reduction along the draft axis. Once every row has its own stopping
position `n[b]`, think about how to gather row `b`'s distributions at position
`n[b]` in a single indexing operation.

---

For `verify_batch`, gather $p_{b,i}[x_{b,i}]$ and $q_{b,i}[x_{b,i}]$ along the
vocabulary axis into `p_x` and `q_x`, compare `u_accept * q_x < p_x`, and take the
cumulative product along the draft axis (`dim=1`). Summing that cumulative product
along `dim=1` gives `n_accepted`. Next, index with two index vectors,
`rows = arange(B)` and `n_accepted`, so `p[rows, n_accepted]` extracts a `(B, V)`
matrix at each row's stopping position. Concatenating a zero slice onto `q` at
position $K$ to form `q_ext` unifies the rejection and bonus cases into
`max(0, p[rows, n_accepted] - q_ext[rows, n_accepted])`. Finally, compute the
cumulative sum `cdf` along the vocabulary axis and compare it against a `(B, 1)`
threshold `u_resample[:, None] * cdf[:, -1:]` so every row uses its own uniform
and its own total.

---

For `advance_batch`, build the output `(B, W)` matrix column by column, where
`W = max(lengths + n_accepted + 1)`. Define `cols = arange(W)[None, :]`,
`start = lengths[:, None]`, and `stop = (lengths + n_accepted)[:, None]`. Column
`c` of row `b` takes:

- the old token from `tokens[b, c]` if `c < start`,
- `draft_tokens[b, c - start]` if `start <= c < stop`,
- `next_tokens[b]` if `c == stop`,
- `pad_id` otherwise.

Gather both the old tokens and the drafts using clamped indices so every index
stays in bounds even where the gathered value is masked out, then combine the four
cases with `where`. Return `lengths + n_accepted` as `cache_lengths`.
