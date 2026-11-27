Write out a single output element, $y_i = \sum_j x_j \,(\text{codes}_{i,j} \cdot \text{scale}_{i,g} + \text{bias}_{i,g})$, where $g = \lfloor j / \text{group\_size} \rfloor$. Within a single group $g$, neither `scale[i, g]` nor `bias[i, g]` depends on $j$. What can you factor out of the inner sum over $j \in g$, and what stays inside?

---

Two sums stay inside group $g$. The first is the dot product $\sum_{j \in g} x_j \,\text{codes}_{i,j}$ between a group of inputs and a group of raw codes. The second is the input sum $\sum_{j \in g} x_j$, which does not depend on the output row $i$. Let $G = \text{in} / \text{group\_size}$. Reshape `x` to `(..., G, group_size)` and `codes` to `(out, G, group_size)` so that group `g` of `x` aligns with group `g` of every row of `codes`.

---

Let `k = group_size`, `xg` be `x` reshaped to `(..., G, k)`, and `cg` be `codes` cast to float32 and reshaped to `(out, G, k)`. Then compute:

```python
dots = einsum("...gk,ogk->...og", xg, cg)   # (..., out, G): inputs times raw codes
sums = xg.sum(-1)                           # (..., G): shared by every output row
y = (dots * scale).sum(-1) + sums @ bias.T  # scale and bias are both (out, G)
```

`dots * scale` broadcasts `(..., out, G)` against `(out, G)`, and `sums @ bias.T` is a standard matrix multiplication of `(..., G)` with `(G, out)`.
