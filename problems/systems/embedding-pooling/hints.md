For each mode, ask which positions of a row the result is allowed to read. Then ask
where those positions are when the padding is on the right, and where they are when
it is on the left. Is "the last column" ever the same thing as "the last real
token"?

---

`"mean"`: multiply the states by the mask (broadcast to `(B, T, 1)`) so the padded
states become zero, sum over `T`, and divide by the number of real tokens, which is
the sum of the mask and not `T`.

`"cls"` and `"last"`: compute one position per row and gather. Build
`positions = arange(T)`. The last real position is the largest position whose mask
is set; the first is the smallest. Both can be written as a `max` or `min` over
`T` after replacing the positions you do not want with a value that cannot win.

---

```python
last = (positions * mask).max(axis=1)  # 0 can never beat a real position
first = where(mask != 0, positions, T).min(axis=1)  # T can never beat one either
pooled = states[arange(B), index]  # (B, D)
```

Normalize with `p / maximum(norm(p, axis=-1, keepdims=True), 1e-12)`; the floor is
what keeps a zero vector from becoming `0 / 0`. `truncate_matryoshka` is the same
normalization applied to `e[..., :dim]`.
