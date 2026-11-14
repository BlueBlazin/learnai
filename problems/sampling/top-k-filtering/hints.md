Comparing each logit with the k-th largest value of its row cannot be the whole
answer: if that value occurs three times and only one place is left, all three pass
the comparison. What do you need to know about each token, other than its value, to
decide whether it is among the first `k`?

---

You need each token's *rank*: its position in the row sorted by decreasing logit, with
equal logits ordered by index. A stable descending argsort gives `order`, where
`order[b, r]` is the token at rank `r`. That is the wrong way around: you want
`rank[b, v]`, the rank of token `v`. The two are inverse permutations of each other.
Handle `k <= 0` and `k >= V` before any of this, by returning the logits.

---

The argsort of a permutation is its inverse, so `rank = argsort(order)`, and the
answer is `where(rank < k, logits, -inf)`. For the stable descending sort: PyTorch
needs `torch.argsort(logits, dim=-1, descending=True, stable=True)`; in JAX and MLX
`argsort` is stable, and sorting `-logits` gives descending order with ties still in
index order (sorting ascending and flipping would reverse the ties). Instead of the
second argsort you can also scatter: write `True` at the positions `order[:, :k]` of
an all-`False` table.
