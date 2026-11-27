Only a `window × P` block of the attention matrix matters. Identify which rows and columns belong to that block, slice it out first, and check the shape left after summing the votes.

---

`vote = attn[:, :, P:, :P].sum(axis=2)` has shape `(B, H, P)`. You can implement stride-1 max-pooling of width `kernel` without a pooling layer. Pad the last axis with `kernel // 2` copies of `-inf` on each side, take the `kernel` shifted slices `padded[..., o : o + P]` for `o = 0 .. kernel - 1`, and take their elementwise maximum. Padding with `-inf` ensures out-of-bounds positions never win. In PyTorch, `F.max_pool1d(vote, kernel, stride=1, padding=kernel // 2)` computes the same result.

---

Rank the pooled votes with a stable descending sort so equal values stay in increasing position order. Use `torch.argsort(pooled, descending=True, stable=True)` in PyTorch, or `argsort(-pooled)` in JAX and MLX. Take the first `budget - window` indices, sort them in ascending order, and concatenate `arange(P, L)` broadcast to `(B, H, window)`. Handle `L <= budget` up front by returning `arange(L)` broadcast to `(B, H, L)`.
