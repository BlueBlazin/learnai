With the zero-point fixed at 0, each row's grid is controlled solely by its scale. If the largest-magnitude weight in a row must land on the largest code, what must that row's scale be? How many codes sit on each side of zero when the grid is symmetric?

---

Take the maximum of $|w|$ along axis 1 to get a `(out,)` vector of row maxima, then divide by $q_{\max} = 2^{\,\text{bits} - 1} - 1$. Replace the scale of any all-zero row with `1.0` before dividing (for example, with `where(max_abs > 0, ...)`).

To divide or multiply a `(out, in)` matrix by a `(out,)` row vector, add a trailing axis so the scale has shape `(out, 1)`. Without it, broadcasting aligns the vector with the columns, which raises a shape error when `out != in` and silently gives the wrong answer when the matrix is square.

---

$$
q = \operatorname{clamp}\!\big(\operatorname{round}(w \,/\, \text{scale}[:, \text{None}]),\ -q_{\max},\ q_{\max}\big)
\quad \text{cast to int8},
\qquad
\hat{w} = \operatorname{float32}(q) \cdot \text{scale}[:, \text{None}].
$$

Round before casting to `int8`, because integer casts truncate toward zero. Check your implementation on a row like `[0.5, -2.0]` with `bits = 8`. The scale should be $2 / 127$, and the codes should be `[32, -127]`, not `[32, -128]`.
