Start with the direct translation of the formula and run the public cases: the mean of
the squares along the last axis (keep that axis, so it broadcasts), plus `eps`, square
root, divide, multiply by `weight`. Which case fails, and what is the largest number
your code computes on the way to an answer that is of order 1?

---

RMSNorm is almost scale-invariant: for any $s > 0$,

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x / s}{\sqrt{\operatorname{mean}\big((x/s)^2\big) + \varepsilon / s^2}} .
$$

So you may divide the vector by any positive number first, as long as you divide
epsilon by its square. Choose $s$ per vector so that no entry of $x / s$ can have a
square that overflows.

---

Take $s = \max(1, \max_j |x_j|)$ along the last axis, with the axis kept. Then every
entry of $u = x / s$ is in $[-1, 1]$, and the result is

$$
\frac{u}{\sqrt{\operatorname{mean}(u^2) + \varepsilon / s^2}} \; g .
$$

The floor of 1 matters: small vectors are left alone, so an all-zero vector does not
divide by zero and $\varepsilon / s^2$ does not blow up. For huge $s$ the term
$\varepsilon / s^2$ rounds to zero, which is harmless, because $\operatorname{mean}(u^2)$
is at least $1/d$ when the largest entry of $u$ has magnitude 1.
