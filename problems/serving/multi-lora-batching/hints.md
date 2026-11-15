Split the formula into the part that depends on the row's adapter and the part that
does not. Which of the two is the big matmul? If you had a tensor holding, for every
row of the batch, the $A$ of that row's adapter, what shape would it have, and which
single indexing operation builds it from `a_stack` and `adapter_ids`?

---

Indexing the stacks with the ids gathers one adapter per row, on the adapter axis:
`a_stack[adapter_ids]` has shape `(B, r_max, in)` and `b_stack[adapter_ids]` has
`(B, out, r_max)`. Two batched matmuls then apply them: `x @ a.transpose` gives
`(B, T, r_max)` and `@ b.transpose` gives `(B, T, out)` (treat a 2-D `x` as `T = 1`).
The padding needs no special handling in the products: padded rows of $A$ produce
zeros, which meet zero columns of $B$. It matters only for the scale, which is
per row: `alpha / ranks[adapter_ids]`.

That leaves the rows with id $-1$. As an index, $-1$ is valid and means "the last
adapter". What can you index with instead, and how do you make that row's update
vanish afterwards?

---

```python
y = base(x)  # once, for the whole batch
has_adapter = adapter_ids >= 0
index = clamp(adapter_ids, min=0)  # -1 -> 0: any valid adapter will do
a, b = a_stack[index], b_stack[index]
scale = where(has_adapter, alpha / ranks[index], 0.0)  # (B,)
tokens = x if x.ndim == 3 else x[:, None, :]  # (B, T, in)
delta = (tokens @ swap(a)) @ swap(b) * scale[:, None, None]  # (B, T, out)
return y + delta.reshape(y.shape)
```

`swap` exchanges the last two axes. The scale must be shaped `(B, 1, 1)` so that it
lines up with the batch axis and not with the token axis.
