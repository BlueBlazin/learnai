Three of the four functions start from the same tensor, the elementwise difference.
What distinguishes them is only how that difference is reduced to one number. For each
one, ask: over which elements does the reduction run, and does the sign of a
difference survive it?

---

`mse` and `max_abs_error` reduce over every element with no axis argument: the mean of
the squared difference, and the maximum of its absolute value.

`sqnr_db` is a ratio of two **powers**: the sum of squares of `ref` over the sum of
squares of `ref - approx`. Decibels of a power ratio are $10 \log_{10}$; the factor 20
belongs to ratios of amplitudes (norms), which are the square roots of these. Do not
guard the division: floating-point division of a positive number by zero is $+\infty$,
and $\log_{10}(+\infty)$ is $+\infty$, which is the specified answer.

`output_error` needs the layer's outputs first. With `w` of shape `(out, in)` and `x`
of shape `(n, in)`, which product has shape `(n, out)`?

---

$$
Y = X W^\top, \qquad \hat{Y} = X \hat{W}^\top, \qquad
\operatorname{output\_error} = \sqrt{\frac{\sum (Y - \hat{Y})^2}{\sum Y^2}} .
$$

In code, `x @ w.T`. Both sums run over the whole `(n, out)` matrix: it is the ratio of
two norms, not an average of per-row ratios, and the denominator is the exact output
`Y`. Check yourself with an approximation that is 10% too large everywhere,
`approx = 1.1 * ref`: the SQNR must be 20 dB, and the output error 0.1.
