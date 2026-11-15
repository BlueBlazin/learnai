RMSNorm divides a vector by a single statistic, the root mean square. LayerNorm uses
two: where is the vector centered, and how spread out is it around that center? Which
of the two must be computed first, and from what?

---

Compute the mean along the last axis and keep the reduced dimension
(`keepdim=True` / `keepdims=True`) so that it broadcasts against `x`. Subtract it. The
variance is then the mean of the *squared deviations*, again along the last axis with
the dimension kept: that is the biased variance, divided by $d$. If you call your
framework's `var` instead, check its default: `torch.var` divides by $d - 1$ unless
you pass `correction=0`.

---

The whole function is three lines:

$$
c = x - \operatorname{mean}(x), \qquad
v = \operatorname{mean}(c^2), \qquad
y = \frac{c}{\sqrt{v + \varepsilon}} \cdot g + b .
$$

Epsilon goes under the root, the gain multiplies before the bias is added, and a
constant vector needs no special case: $c = 0$, so the result is $0 / \sqrt{\varepsilon}
\cdot g + b = b$.
