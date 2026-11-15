A grid with three values is described by one number: where the border between "small
enough to be 0" and "large enough to be $\pm 1$" lies. If the grid points are
$-s$, $0$, and $s$, which weights are closer to $0$ than to $\pm s$? And which single
statistic of the matrix puts a useful share of the weights on each side of that
border: the largest magnitude, or the typical one?

---

The scale is the mean of $|w|$ over the whole matrix, one number, with a `where` that
turns a mean of 0 into 1 so that the division is safe. Then
`round(w / scale)` followed by a clamp to $[-1, 1]$ and a cast to int8. Use the
framework's `round`: it sends $\pm 0.5$ to $0$, as the statement requires, where
`floor(r + 0.5)` would send $0.5$ to $1$.

For the layer, add a batch-side axis to `x` so that it has shape `(..., 1, in)` and
broadcasts against `t` of shape `(out, in)`.

---

Select with `where`, then sum over the input axis:

$$
\text{plus} = \sum_j \operatorname{where}(t = 1,\ x,\ 0), \qquad
\text{minus} = \sum_j \operatorname{where}(t = -1,\ x,\ 0), \qquad
y = \text{scale} \cdot (\text{plus} - \text{minus}).
$$

Compare `t` with $1$ and with $-1$ explicitly. A test such as `t >= 0` for the
positive set also selects the inputs whose weight is $0$.
