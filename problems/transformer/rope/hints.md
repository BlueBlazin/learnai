Work out the tensor shapes before writing the trigonometry. You need one angle per
`(token, pair)`, which is a table of shape `(T, D/2)` built from `positions` of shape
`(T,)` and the frequencies $\theta$ of shape `(D/2,)`. How can you extract the two
members of every pair from `x` as tensors of shape `(B, H, T, D/2)` so the angle table
broadcasts against them?

---

Break the calculation into three parts:

- **Frequencies.** The exponents are $-2i/D$ for $i = 0, \dots, D/2 - 1$, which gives
  $0, -2/D, -4/D, \dots$. Compute this by taking `arange(0, D, 2)`, dividing by `D`,
  and negating.
- **Angles.** Cast `positions` to float32, reshape it as a column `(T, 1)`, and
  multiply by $\theta$ of shape `(D/2,)`.
- **Pairs.** The even-indexed entries `x[..., 0::2]` are the first elements of each
  pair, and the odd-indexed entries `x[..., 1::2]` are the second elements.

Apply the 2D rotation to those two halves. Then place the results back so that the
first rotated pair occupies output indices 0 and 1, the second occupies 2 and 3, and so
on.

---

With `x1 = x[..., 0::2]`, `x2 = x[..., 1::2]`, and `cos` and `sin` of the angle table,
compute

$$
y_1 = x_1 \cos - x_2 \sin, \qquad y_2 = x_1 \sin + x_2 \cos .
$$

To interleave them again, stack $y_1$ and $y_2$ along a **new last axis** to get shape
`(B, H, T, D/2, 2)`, then reshape to `(B, H, T, D)`. Reshaping reads the last axis
fastest, emitting $y_1[0], y_2[0], y_1[1], y_2[1], \dots$ in the exact adjacent-pair
layout. Concatenating $y_1$ and $y_2$ along the existing last axis would put them in
the wrong order.
