Split the formula into the part that depends on each row's adapter and the part that does not. Which part is the expensive matmul? If you build a tensor holding each row's $A$ matrix, what shape does it have, and which single indexing operation builds it from `a_stack` and `adapter_ids`?

---

Indexing the weight stacks with `adapter_ids` gathers one adapter per row along the batch axis. `a = a_stack[adapter_ids]` has shape `(B, r_max, in)` and `b = b_stack[adapter_ids]` has shape `(B, out, r_max)`. Two batched matmuls then apply them: `x @ a.transpose` gives `(B, T, r_max)`, and multiplying by `b.transpose` gives `(B, T, out)` (treat a 2D `x` as `T = 1`). Zero-padding needs no special handling in the products because padded rows of $A$ produce zeros that multiply zero columns of $B$. The true rank matters only for the per-row scale `alpha / ranks[adapter_ids]`.

That leaves rows with adapter ID `-1`. As a tensor index, `-1` is valid and selects the last adapter. What safe index can you substitute, and how do you zero out that row's update afterward?

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

Here `swap` transposes the last two axes. Reshape `scale` to `(B, 1, 1)` so it broadcasts over the batch axis rather than the token axis.
