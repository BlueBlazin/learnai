Which tokens belong in the nucleus depends on their rank, not their vocabulary index.
After sorting each row's probabilities in decreasing order, the nucleus is a prefix of
the sorted row. What single quantity, computed for every rank at once, tells you whether
that rank falls inside the prefix?

---

Take the cumulative sum of the sorted probabilities. For the token at rank $k$, compare
`p` against the mass of the tokens ranked *before* it ($c_{k-1}$), not the mass
including it ($c_k$). Check your rule on the worked example. The token that crosses `p`
must be kept, and the top token must survive even when its own probability exceeds `p`.
Use a stable sort so tied probabilities stay in index order.

---

The mass before each rank is `cumsum(sorted) - sorted`, and the keep condition is
`mass_before < p`. Zero out the remaining entries, divide each row by its sum, and then
invert the sort. Because `order[b, k]` gives the token index at rank `k`, the kept value
at rank `k` belongs at position `order[b, k]`. Use a **scatter** along the last axis
(`scatter` in PyTorch, `.at[rows, order].set(...)` in JAX, or `put_along_axis` in MLX).
Calling `gather` with `order` applies the sort a second time instead of unsorting.
