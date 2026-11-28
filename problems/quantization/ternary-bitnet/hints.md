A three-value grid with points $-s$, $0$, and $s$ is controlled by one number: the border between values small enough to become $0$ and values large enough to become $\pm 1$. Which weights are closer to $0$ than to $\pm s$? And which matrix statistic puts a useful fraction of weights on both sides of that border: the maximum magnitude, or the mean magnitude?

---

Compute `scale` as the mean of $|w|$ across the whole matrix, using `where` to replace a mean of $0$ with $1$ so the division is safe. Then compute `round(w / scale)`, clamp to $[-1, 1]$, and cast to `int8`. Use your framework's built-in `round`, which rounds $\pm 0.5$ to $0$, whereas `floor(r + 0.5)` rounds $0.5$ to $1$.

For `ternary_matmul`, insert a size-1 axis into `x` so its shape becomes `(..., 1, in)` and broadcasts against `t` of shape `(out, in)`.

---

Select the positive and negative contributions with `where`, then sum over the input axis:

$$
\text{plus} = \sum_j \operatorname{where}(t = 1,\ x,\ 0), \qquad
\text{minus} = \sum_j \operatorname{where}(t = -1,\ x,\ 0), \qquad
y = \text{scale} \cdot (\text{plus} - \text{minus}).
$$

Check `t == 1` and `t == -1` explicitly. Using `t >= 0` for the positive set mistakenly includes inputs whose weight is $0$.
