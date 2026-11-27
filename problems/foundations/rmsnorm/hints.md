Start with the direct translation of the formula and run the public cases. Take the mean
of the squares along the last axis (keeping that axis so it broadcasts), add `eps`, take
the square root, divide `x` by it, and multiply by `weight`. Which case fails, and what
is the largest intermediate value your code computes on the way to an answer of order 1?

---

RMSNorm is almost scale-invariant. For any $s > 0$:

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x / s}{\sqrt{\operatorname{mean}\big((x/s)^2\big) + \varepsilon / s^2}} .
$$

You can divide each vector by a positive constant $s$ before squaring, as long as you
also divide $\varepsilon$ by $s^2$. Choose $s$ per vector so that no entry of $x / s$ can
overflow when squared.

---

Take $s = \max(1, \max_j |x_j|)$ along the last axis, keeping that axis. Then every entry
of $u = x / s$ lies in $[-1, 1]$, and the result is:

$$
\frac{u}{\sqrt{\operatorname{mean}(u^2) + \varepsilon / s^2}} \; g .
$$

The floor of 1 is important. It leaves small vectors untouched, so an all-zero vector
does not divide by zero and $\varepsilon / s^2$ does not blow up. For huge $s$,
$\varepsilon / s^2$ rounds to `0`, which is harmless because $\operatorname{mean}(u^2)$
is at least $1/d$ (where $d$ is the vector length) when the largest entry of $u$ has
magnitude 1.
