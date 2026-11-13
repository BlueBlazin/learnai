Which tokens are in the nucleus depends on their rank, not their position. After
sorting each row's probabilities in decreasing order, the nucleus is a prefix of the
sorted row. What single quantity, computed for every rank at once, tells you whether
that rank is inside the prefix?

---

Take the cumulative sum of the sorted probabilities. Be careful about which cumulative
sum decides: for the token at rank $k$, compare `p` with the mass of the tokens
*before* it ($c_{k-1}$), not with the mass including it ($c_k$). Check your rule on the
worked example: the token that crosses `p` must be kept, and the top token must
survive even when it alone has more than `p`. Use a stable sort so that tied
probabilities stay in index order.

---

The mass before each rank is `cumsum(sorted) - sorted`, and the keep rule is
`mass_before < p`. Zero the other entries, divide each row by its sum, and then undo
the sort. The sort gave you `order`, with `order[b, k]` = the token at rank `k`, so
the value at rank `k` belongs at position `order[b, k]`: that is a **scatter** along
the last axis (`scatter` in PyTorch, `.at[rows, order].set(...)` in JAX,
`put_along_axis` in MLX). A `gather` with `order` sorts a second time instead of
unsorting.
