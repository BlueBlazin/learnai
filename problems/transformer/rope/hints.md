Work out the shapes before the trigonometry. You need one angle per (token, pair):
a table of shape `(T, D/2)` built from `positions` of shape `(T,)` and the frequencies
$\theta$ of shape `(D/2,)`. How do you take the two members of every pair out of `x`
as tensors of shape `(B, H, T, D/2)`, so that the angle table broadcasts against them?

---

- Frequencies: the exponents are $-2i/D$ for $i = 0, \dots, D/2 - 1$, that is
  $0, -2/D, -4/D, \dots$: an `arange(0, D, 2)` divided by `D`, negated.
- Angles: `positions` (converted to float32) as a column times $\theta$ as a row.
- Pairs: the even-indexed entries `x[..., 0::2]` are the first members and the
  odd-indexed entries `x[..., 1::2]` are the second members.

Apply the 2-D rotation to the two halves. The remaining problem is to put the results
back so that the first rotated pair occupies output indices 0 and 1, the second 2 and
3, and so on.

---

With `x1 = x[..., 0::2]`, `x2 = x[..., 1::2]`, and `cos`, `sin` of the angle table:

$$
y_1 = x_1 \cos - x_2 \sin, \qquad y_2 = x_1 \sin + x_2 \cos .
$$

To interleave them again, stack $y_1$ and $y_2$ on a **new last axis**, giving shape
`(B, H, T, D/2, 2)`, and reshape to `(B, H, T, D)`. The reshape reads the last axis
fastest, so it emits $y_1[0], y_2[0], y_1[1], y_2[1], \dots$, which is exactly the
adjacent-pair layout. Concatenating $y_1$ and $y_2$ along the existing last axis
would produce the wrong order.
