RMSNorm divides a vector by a single statistic, the root mean square. LayerNorm uses two statistics: where the vector is centered, and how spread out it is around that center. Which of the two must be computed first, and from what?

---

Compute the mean along the last axis and keep the reduced dimension (`keepdim=True` or `keepdims=True`) so it broadcasts against `x`, then subtract it from `x`. The biased variance (divided by $d$) is the mean of the *squared deviations* along the last axis, again keeping the dimension. If you use your framework's `var` instead, check its default, since `torch.var` divides by $d - 1$ unless you pass `correction=0`.

---

The whole function is three lines:

$$
c = x - \operatorname{mean}(x), \qquad
v = \operatorname{mean}(c^2), \qquad
y = \frac{c}{\sqrt{v + \varepsilon}} \cdot g + b .
$$

Place $\varepsilon$ inside the square root and multiply by the gain $g$ before adding the bias $b$. A constant vector needs no special case because $c = 0$ gives $0 / \sqrt{\varepsilon} \cdot g + b = b$.
