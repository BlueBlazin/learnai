Pull the maximum outside the logarithm:

$$
\operatorname{LSE}(x) = m + \log \sum_i e^{x_i - m}, \qquad m = \max_i x_i .
$$

The sum now contains one term equal to $1$ and the rest in $[0, 1]$, so it lies in $[1, n]$ and its logarithm in $[0, \log n]$. That is also the proof of the bounds property.

The only trap is the empty distribution. When every entry is $-\infty$, $x_i - m$ is $-\infty - (-\infty)$, which is `nan`. Guard the shift, not the result.
