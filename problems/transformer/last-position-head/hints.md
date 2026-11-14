The function is two operations, a selection and a matrix product, and they can be done
in either order with the same result. Count the multiplications each order performs
for `B = 8`, `T = 512`, `D = 64`, `V = 8192`. Which order never creates a tensor with
both a `T` axis and a `V` axis?

---

Select first. Take the hidden state of the last position of every sequence, a tensor
of shape `(B, D)`, and only then apply the output matrix. Two details:

- The time axis is the **second** axis of `hidden`. Indexing the first axis picks a
  sequence, not a position.
- Index with an integer, not with a slice of length one, so that the time axis
  disappears and the result is `(B, V)` and not `(B, 1, V)`.

The matrix is stored as `(V, D)`, so mapping row vectors of width `D` to width `V`
needs its transpose.

---

```python
last = hidden[:, -1]  # (B, D): the last position of each sequence
logits = last @ w_out.T  # (B, V)
```

`-1` as an index means "the last one" in every framework, for any `T`, including
`T = 1`. If your result is right but the speed case fails, look for a product whose
left operand still has shape `(B, T, D)`.
