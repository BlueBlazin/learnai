Write out one output, $y_i = \sum_j x_j \,(\text{codes}_{i,j} \cdot \text{scale}_{i,g}
+ \text{bias}_{i,g})$, and look at a single group. Within a group the scale and the
bias do not depend on $j$. What can you pull out of the sum over $j$, and what is left
inside?

---

Two things are left inside, one per term: $\sum_{j \in g} x_j \,\text{codes}_{i,j}$, a
dot product of a slice of inputs with a slice of raw codes, and $\sum_{j \in g} x_j$,
which does not involve the row $i$ at all. To get them for all groups at once, reshape
`x` to `(..., G, group_size)` and `codes` to `(out, G, group_size)`: the same reshape
on both, so that group `g` of `x` meets group `g` of every row of codes.

---

With `xg` of shape `(..., G, k)` and the codes as float `cg` of shape `(out, G, k)`:

```python
dots = einsum("...gk,ogk->...og", xg, cg)   # (..., out, G): inputs times raw codes
sums = xg.sum(-1)                           # (..., G): shared by every output row
y = (dots * scale).sum(-1) + sums @ bias.T  # scale and bias are both (out, G)
```

`dots * scale` broadcasts `(..., out, G)` against `(out, G)`, and `sums @ bias.T` is an
ordinary matrix product of `(..., G)` with `(G, out)`.
