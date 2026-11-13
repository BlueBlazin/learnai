Softmax does not change when you add the same constant to every logit along the axis:
$\operatorname{softmax}(x) = \operatorname{softmax}(x - c)$. The constant cancels
between the numerator and the denominator. Which choice of $c$ guarantees that no
exponential can overflow?

---

Subtract the maximum along `axis`. After the shift the largest entry is exactly `0`,
so every exponent is at most `0` and every exponential lies in $[0, 1]$: no overflow.
The largest term contributes $e^0 = 1$ to the sum, so the denominator is at least `1`:
no division by zero. Keep the reduced dimension (`keepdim=True` / `keepdims=True`) so
that the maximum broadcasts against `x` along the right axis.

---

For `log_softmax`, never take the log of a probability that may have underflowed. Stay
in log space instead:

$$
\operatorname{log\_softmax}(x)_i = (x_i - m) - \log \sum_j e^{x_j - m},
\qquad m = \max_j x_j .
$$

The sum is between `1` and `n`, so its logarithm is always finite, and the result is
the shifted logit minus a small non-negative number.
