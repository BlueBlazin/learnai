The overflow happens in $x_j^2$, before the mean is taken, so no clever way of taking
the mean helps. RMSNorm has a property that does: what happens to
$x / \sqrt{\operatorname{mean}(x^2)}$ when you divide every entry of $x$ by the same
positive number $c$ first? And for which $c$ is every square guaranteed to fit?

---

Divide by the largest magnitude in the vector, $c = \max_j |x_j|$. Then $u = x / c$ has
all entries in $[-1, 1]$, the largest is exactly $\pm 1$, and $\operatorname{mean}(u^2)$
lies in $[1/d, 1]$: no overflow, and no underflow that matters. Epsilon must be
rescaled with it:

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{u}{\sqrt{\operatorname{mean}(u^2) + \varepsilon / c^2}} .
$$

Now look at $\varepsilon / c^2$ in float16 for a tiny vector, and at $c$ for an
all-zero one.

---

Write the rescaled epsilon as $(\sqrt{\varepsilon} / c)^2$, with $\sqrt{\varepsilon}$
computed once as a Python float (`math.sqrt(eps)`, about $3 \times 10^{-3}$: an
ordinary float16, unlike $\varepsilon$ itself). It overflows only when $c$ is far below
$\sqrt{\varepsilon}$, and in that regime the vector is negligible next to epsilon
anyway. So bound the scale from below:

$$
c = \max\!\big(\max_j |x_j|,\ \sqrt{\varepsilon}\big), \qquad
u = \frac{x}{c}, \qquad r = \frac{\sqrt{\varepsilon}}{c}, \qquad
y = \frac{u}{\sqrt{\operatorname{mean}(u^2) + r^2}} \cdot g .
$$

Both $|u_j|$ and $r$ are at most 1, one of the two terms under the root is at least
$1/d$, and $c$ is never zero, so the all-zero vector needs no special case.
