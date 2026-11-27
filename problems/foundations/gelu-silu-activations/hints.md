Start with exact GELU. It multiplies $x$ by the standard normal CDF, which is a shifted and rescaled `erf`. What argument does `erf` take? You can verify your scaling at $x = 1$, where $\Phi(1) = 0.8413$ (about 84 percent of a normal distribution lies below one standard deviation).

---

Implement the `tanh` approximation as a separate branch selected by `approximate`. Match the constants carefully. The factor in front of the cubic is $\sqrt{2/\pi} \approx 0.79788$ (with the square root covering the entire fraction), and the cubic coefficient is `0.044715`. If your `"none"` and `"tanh"` outputs agree to six decimal places, one branch is returning the other's formula.

For SiLU, consider the two textbook spellings of the sigmoid, $1 / (1 + e^{-x})$ and $e^{x} / (1 + e^{x})$. Evaluate both by hand in float32 at $x = 100$ and $x = -100$, keeping in mind that $e^{100}$ overflows to `inf`.

---

The three formulas are:

$$
\frac{x}{2}\Big(1 + \operatorname{erf}\big(x / \sqrt{2}\big)\Big), \qquad
\frac{x}{2}\Big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\Big), \qquad
\frac{x}{1 + e^{-x}} .
$$

The SiLU formula is safe at both extremes. For large positive $x$, $e^{-x}$ underflows to `0.0` and the expression evaluates to $x / 1$. For large negative $x$, $e^{-x}$ overflows to `inf` and $x / \infty$ evaluates to `-0.0`, which matches the limit. Neither `erf` nor `tanh` overflows, and $x^3$ reaches at most $10^{12}$ under the problem constraints, so both GELU formulas stay finite without extra guards.
