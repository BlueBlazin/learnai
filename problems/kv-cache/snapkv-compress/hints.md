Only a `window × P` block of the attention matrix matters: which rows and which
columns? Slice it out first, and write down the shape of what is left after the
votes are summed.

---

`vote = attn[:, :, P:, :P].sum(axis=2)` has shape `(B, H, P)`. Max-pooling with
stride 1 and width `kernel` can be written without a pooling layer: pad the last
axis with `kernel // 2` entries of `-inf` on each side, take the `kernel` shifted
views `padded[..., o : o + P]` for `o = 0 .. kernel - 1`, and reduce them with an
elementwise maximum. The `-inf` padding is what keeps the ends honest. (In PyTorch,
`F.max_pool1d(vote, kernel, stride=1, padding=kernel // 2)` computes the same
thing.)

---

Rank the pooled votes with a stable descending sort, so that equal values stay in
position order: `torch.argsort(pooled, descending=True, stable=True)`, or
`argsort(-pooled)` in JAX and MLX. Take the first `budget - window` positions of the
ranking, sort them, and append `arange(P, L)` broadcast to `(B, H, window)`. Handle
`L <= budget` before any of this by returning `arange(L)` for every head.
