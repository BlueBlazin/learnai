Comparing each logit against the k-th largest value in its row is not enough. If that boundary value appears three times and only one slot remains, all three entries pass the comparison. What do you need to know about each token, besides its logit value, to tell whether it belongs in the first `k`?

---

You need each token's *rank*, meaning its position when the row is sorted by decreasing logit with ties ordered by lowest index. A stable descending `argsort` produces `order`, where `order[b, r]` is the token at rank `r`. You need the inverse mapping `rank[b, v]`, which gives the rank of token `v`. Handle `k <= 0` and `k >= V` up front by returning `logits` unchanged.

---

The `argsort` of a permutation is its inverse, so `rank = argsort(order)`, and the filtered output is `where(rank < k, logits, -inf)`. For the stable descending sort, PyTorch uses `torch.argsort(logits, dim=-1, descending=True, stable=True)`. In JAX and MLX, `argsort` is stable, so sorting `-logits` gives descending order while keeping ties in index order (sorting ascending and flipping would reverse ties). To avoid the second `argsort`, you can alternatively scatter `True` into an all-`False` boolean mask at positions `order[:, :k]`.
