Three of the four functions start from the elementwise difference between the two input tensors and differ only in how they reduce it to a single scalar. For each one, ask which elements the reduction should cover and whether the sign of a difference should matter.

---

`mse` and `max_abs_error` reduce over every element with no axis argument. Compute the mean of the squared differences for `mse`, and the maximum of the absolute differences for `max_abs_error`.

`sqnr_db` is a ratio of two **powers**: the sum of squares of `ref` divided by the sum of squares of `ref - approx`. Decibels for a power ratio use $10 \log_{10}$, whereas $20 \log_{10}$ applies only to amplitude (norm) ratios. Do not add an epsilon to the denominator. In IEEE floating-point arithmetic, dividing a positive number by zero gives $+\infty$, and $\log_{10}(+\infty)$ gives $+\infty$, which matches the specification.

For `output_error`, compute the exact and approximate layer outputs first. Given `x` with shape `(n, in)` and `w` with shape `(out, in)`, which matrix product produces shape `(n, out)`?

---

$$
Y = X W^\top, \qquad \hat{Y} = X \hat{W}^\top, \qquad
\operatorname{output\_error} = \sqrt{\frac{\sum (Y - \hat{Y})^2}{\sum Y^2}} .
$$

In code, compute `y = x @ w.T` and `y_hat = x @ w_hat.T`. Both sums run over the entire `(n, out)` matrix, so the result is a single ratio of Frobenius norms rather than an average of per-row ratios, and the denominator uses the exact output `Y`. As a sanity check, test with `approx = 1.1 * ref` (a 10% error everywhere). Your `sqnr_db` should return 20 dB and your `output_error` should return 0.1.
