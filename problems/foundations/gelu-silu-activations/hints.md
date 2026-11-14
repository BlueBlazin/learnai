Start with the exact GELU: it is $x$ times the normal CDF, and the normal CDF is a
shifted and rescaled `erf`. What is the argument of the `erf`? Check your answer on
$x = 1$: $\Phi(1) = 0.8413$, the familiar "84 percent of a normal distribution lies
below one standard deviation".

---

The tanh form is a separate branch, selected by the string `approximate`, and it must
match the formula digit for digit: the constant in front of the cubic is
$\sqrt{2/\pi} = 0.79788$ (the square root covers the whole fraction), and the
coefficient is $0.044715$. If your `"none"` and `"tanh"` results agree to six decimal
places, one branch is returning the other's formula.

For SiLU you need a sigmoid, and there are two textbook spellings:
$1 / (1 + e^{-x})$ and $e^{x} / (1 + e^{x})$. Work out by hand what each one computes
in float32 for $x = 100$ and for $x = -100$, remembering that $e^{100}$ is `inf`.

---

The three lines:

$$
\frac{x}{2}\Big(1 + \operatorname{erf}\big(x / \sqrt{2}\big)\Big), \qquad
\frac{x}{2}\Big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\Big), \qquad
\frac{x}{1 + e^{-x}} .
$$

The last one is safe at both ends. For large positive $x$, $e^{-x}$ underflows to 0 and
the result is $x / 1$. For large negative $x$, $e^{-x}$ overflows to `inf` and
$x / \infty$ is `-0.0`, which is the right limit. Neither `erf` nor `tanh` can
overflow, and $x^3$ is at most $10^{12}$ here, so the GELU lines need no special care.
