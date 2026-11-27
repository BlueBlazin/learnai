Softmax does not change when you subtract the same constant $c$ from every logit along
the axis, because $e^{-c}$ cancels between the numerator and the denominator:
$\operatorname{softmax}(x) = \operatorname{softmax}(x - c)$. Which choice of $c$
guarantees that no exponential can overflow?

---

Subtract the maximum along `axis`. After the shift, the largest entry is `0`, so every
exponent is at most `0` and every exponential lies in $[0, 1]$, preventing overflow. The
largest term contributes $e^0 = 1$ to the sum, so the denominator is at least `1`,
preventing division by zero. Keep the reduced dimension (`keepdim=True` or
`keepdims=True`) so the maximum broadcasts against `x` along the right axis.

---

For `log_softmax`, never take the log of a probability that may have underflowed. Stay
in log space instead:

$$
\operatorname{log\_softmax}(x)_i = (x_i - m) - \log \sum_j e^{x_j - m},
\qquad m = \max_j x_j .
$$

For a slice of length $n$, the sum lies between 1 and $n$, so its logarithm is always
finite and the result is the shifted logit minus a small non-negative number.
