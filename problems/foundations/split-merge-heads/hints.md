Think about memory order. A reshape keeps elements in the same flat order (last axis fastest) and only redraws axis boundaries. In a `(B, T, D)` tensor, which elements sit next to each other in memory: the features of one token, or the tokens of one head? Which of those two must be adjacent in the target shape `(B, H, T, d)`?

---

Break `split_heads` into two steps and track the shape after each one.

1. Split the last axis from `(B, T, D)` into `(B, T, H, d)`. This is a pure reshape, because the features belonging to head `h` are already contiguous. Element `[b, t, h, i]` in the new shape is element `[b, t, h * d + i]` in the original tensor.
2. Swap the `T` and `H` axes to go from `(B, T, H, d)` to `(B, H, T, d)`. This changes element order and requires a transpose, not a reshape.

To implement `merge_heads`, undo those two steps in reverse order.

---

With `d = D // n_heads`:

```
split:  reshape to (B, T, H, d), then swap axes 1 and 2
merge:  swap axes 1 and 2, then reshape to (B, T, H * d)
```

In PyTorch, use `reshape` (or `contiguous().view`) for the final step of `merge_heads`, not `view` alone. After the transpose, the elements are no longer stored in the contiguous order that the merged shape requires, so `view` raises an error instead of copying.
