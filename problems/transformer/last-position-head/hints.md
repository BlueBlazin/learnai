The function combines two operations, a position selection and a matrix multiplication, which give the same result in either order. Count the multiplications each order performs for `B = 8`, `T = 512`, `D = 64`, `V = 8192`. Which order avoids ever creating a tensor with both a `T` axis and a `V` axis?

---

Select the last position first to obtain a tensor of shape `(B, D)`, and then apply `w_out`. Keep two indexing details in mind:

- The time dimension is the **second** axis of `hidden` (axis 1). Indexing axis 0 selects a sequence instead of a position.
- Use an integer index rather than a length-1 slice so the time axis is removed and the output has shape `(B, V)` instead of `(B, 1, V)`.

Because `w_out` has shape `(V, D)`, transpose it before multiplying on the right of a `(B, D)` matrix.

---

```python
last = hidden[:, -1]  # (B, D): the last position of each sequence
logits = last @ w_out.T  # (B, V)
```

Index `-1` selects the last element in every framework for any $T \ge 1$, including `T = 1`. If your numerical outputs match the reference but the speed test fails, check whether your matrix multiplication still takes a `(B, T, D)` tensor on the left.
