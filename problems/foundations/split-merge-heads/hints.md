Think about memory order. A reshape keeps the elements in the same order (last axis
fastest) and only changes where the axis boundaries are drawn. In a `(B, T, D)` tensor,
which elements are next to each other in that order: the features of one token, or the
tokens of one head? Which of the two does the target shape `(B, H, T, d)` need to be
adjacent?

---

Split in two steps, and name the shape after each one.

1. Cut the last axis in two: `(B, T, D)` to `(B, T, H, d)`. This is a pure reshape. It
   is valid because the features of head `h` are already adjacent: element
   `[b, t, h, i]` of the new shape is element `[b, t, h * d + i]` of the old one.
2. Exchange the `T` and `H` axes: `(B, T, H, d)` to `(B, H, T, d)`. This is a
   transpose, and it cannot be done by a reshape.

To merge, undo the steps in the reverse order.

---

With `d = D // n_heads`:

```
split:  reshape to (B, T, H, d), then swap axes 1 and 2
merge:  swap axes 1 and 2, then reshape to (B, T, H * d)
```

In PyTorch, do the final step of the merge with `reshape` (or `contiguous().view`),
not `view` alone: after a transpose the elements are no longer stored in the order
that the new shape describes, and `view` refuses to copy.
