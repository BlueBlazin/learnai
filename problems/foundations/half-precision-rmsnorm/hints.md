The overflow happens in $x_j^2$ before the mean is taken, so changing how you average
does not fix it. Instead, look at scale invariance. What happens to
$x / \sqrt{\operatorname{mean}(x^2)}$ if you divide every entry of $x$ by the same
positive number $c$ first, and which $c$ guarantees that every square fits in `float16`?

---

Divide each vector of length $d$ by its largest magnitude, $c = \max_j |x_j|$. Then
$u = x / c$ has all entries in $[-1, 1]$ with the largest equal to $\pm 1$, so
$\operatorname{mean}(u^2)$ lies in $[1/d, 1]$. That prevents both overflow and harmful
underflow. Remember to rescale epsilon by the same factor:

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{u}{\sqrt{\operatorname{mean}(u^2) + \varepsilon / c^2}} .
$$

Now check what happens to $\varepsilon / c^2$ in `float16` when the vector is tiny, and
what happens to $c$ when the vector is all zeros.

---

Rewrite the rescaled epsilon as $(\sqrt{\varepsilon} / c)^2$, computing
$\sqrt{\varepsilon}$ once as a Python float with `math.sqrt(eps)`. For example,
$\sqrt{10^{-5}} \approx 3.16 \times 10^{-3}$ is a normal `float16`, unlike $\varepsilon$
itself. The ratio $\sqrt{\varepsilon} / c$ only grows large when $c$ is much smaller than
$\sqrt{\varepsilon}$, where the vector is negligible compared to epsilon anyway. Bound
the scale from below by $\sqrt{\varepsilon}$ and multiply by the gain $g$ (`weight`):

$$
c = \max\!\big(\max_j |x_j|,\ \sqrt{\varepsilon}\big), \qquad
u = \frac{x}{c}, \qquad r = \frac{\sqrt{\varepsilon}}{c}, \qquad
y = \frac{u}{\sqrt{\operatorname{mean}(u^2) + r^2}} \cdot g .
$$

Every $|u_j|$ and $r$ is at most 1, at least one term under the square root is at least
$1/d$, and $c$ is never zero, so the all-zero vector needs no special case.
