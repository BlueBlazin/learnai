With the zero-point fixed at 0, a grid is described by one number, its scale. If the
element of largest magnitude in a row must land exactly on the largest code, what does
the scale of that row have to be? And how many codes are there on each side of zero if
both sides must have the same number?

---

Take the maximum of $|w|$ along axis 1 to get one number per row, shape `(out,)`, and
divide by $q_{\max} = 2^{\,\text{bits}-1} - 1$. Replace the scale of an all-zero row by
1 before dividing by it (a `where` on `max_abs > 0` does it without a branch).

To divide the rows of a `(out, in)` matrix by a `(out,)` vector, the vector needs a
trailing axis of length 1. Without it, broadcasting aligns the vector with the last
axis, the columns: an error when `out != in`, and a silently wrong answer when the
matrix is square.

---

$$
q = \operatorname{clamp}\big(\operatorname{round}(w \,/\, \text{scale}[:, \text{None}]),\ -q_{\max},\ q_{\max}\big)
\quad \text{cast to int8},
\qquad
\hat{w} = \operatorname{float32}(q) \cdot \text{scale}[:, \text{None}] .
$$

Round before the cast, since a cast to an integer type truncates. Check yourself on a
row such as `[0.5, -2.0]` with 8 bits: the scale is $2 / 127$, and the codes are
`[32, -127]`, not `[32, -128]`.
