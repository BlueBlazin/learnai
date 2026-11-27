Implement four standard quantization error metrics: mean squared error (`mse`), signal-to-quantization-noise ratio in decibels (`sqnr_db`), maximum absolute error (`max_abs_error`), and the relative output error of a linear layer (`output_error`).

## Specification

```python
def mse(a, b): ...
def sqnr_db(ref, approx): ...
def max_abs_error(a, b): ...
def output_error(w, w_hat, x): ...
```

Every function reduces over the entire input and returns a **0-d float32 tensor** (a single scalar). In the formulas below, $N$ is the total number of elements in the tensor, and all sums, means, and maxima run over all $N$ elements.

**`mse(a, b)`** computes the mean squared error between `a` and `b`:

$$
\operatorname{mse}(a, b) = \frac{1}{N} \sum_{k} (a_k - b_k)^2 .
$$

**`sqnr_db(ref, approx)`** computes the signal-to-quantization-noise ratio in decibels. The signal power is the sum of squared entries in `ref`, and the noise power is the sum of squared differences between `ref` and `approx`. SQNR expresses their ratio on a decibel scale:

$$
\operatorname{sqnr\_db}(\text{ref}, \text{approx}) =
10 \log_{10} \frac{\sum_k \text{ref}_k^2}{\sum_k (\text{ref}_k - \text{approx}_k)^2} .
$$

Argument order matters because only `ref` appears in the numerator. If `approx` equals `ref` at every element, the noise power is 0 and the result must be $+\infty$ (not a large finite number or NaN). You may assume `ref` is never all zeros.

**`max_abs_error(a, b)`** computes the largest absolute difference at any single element:

$$
\operatorname{max\_abs\_error}(a, b) = \max_k\, |a_k - b_k| .
$$

**`output_error(w, w_hat, x)`** computes the relative error at the output of a linear layer. Here `w` is the exact weight matrix of shape `(out, in)`, `w_hat` is its quantized approximation of the same shape, and `x` is a batch of `n` input rows of shape `(n, in)`. The exact layer output is $Y = X W^\top$ and the approximate output is $\hat{Y} = X \hat{W}^\top$, both of shape `(n, out)`. The relative output error is the ratio of Frobenius norms:

$$
\operatorname{output\_error}(W, \hat{W}, X) =
\frac{\lVert Y - \hat{Y} \rVert_F}{\lVert Y \rVert_F},
\qquad \lVert A \rVert_F = \sqrt{\sum_{i,j} A_{ij}^2} .
$$

Both Frobenius norms run over all $n \cdot \text{out}$ entries. The denominator is the norm of the **exact** output $Y$, which is never all zeros.

Write the arithmetic directly without calling a built-in mean-squared-error loss (such as `mse_loss`). Elementwise operations, reductions (`sum`, `mean`, `max`), `log10`, `sqrt`, and matrix multiplication are all you need.

## Shapes

| name | shape | dtype |
|---|---|---|
| `a`, `b` | the same shape, one to four dimensions | float32 |
| `ref`, `approx` | the same shape, one to four dimensions | float32 |
| `w`, `w_hat` | `(out, in)` | float32 |
| `x` | `(n, in)` | float32 |
| every result | `()` | float32 |

## Example

```python
a = [1.0, -2.0, 3.0, 0.5]
b = [1.1, -2.0, 2.8, 0.5]

mse(a, b)  # 0.0125
sqnr_db(a, b)  # 24.5484
max_abs_error(a, b)  # 0.2
```

The elementwise differences `a - b` are $[-0.1, 0, 0.2, 0]$, and their squares are $[0.01, 0, 0.04, 0]$. The mean of those squares is $0.05 / 4 = 0.0125$, and the largest absolute difference is $0.2$. The signal power is $1 + 4 + 9 + 0.25 = 14.25$ and the noise power is $0.05$, giving a ratio of $285$ and an SQNR of $10 \log_{10} 285 = 24.55$ dB.

```python
w = [[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]]  # (out, in) = (2, 3)
w_hat = [[1.0, 2.25, -1.0], [0.5, -0.5, 1.75]]
x = [[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]]  # n = 4

output_error(w, w_hat, x)  # 0.0905357
```

Multiplying `x` by `w.T` and `w_hat.T` gives:

$$
Y = X W^\top = \begin{bmatrix} 0 & 2.5 \\ 4 & 0.5 \\ -3 & 2.5 \\ 2 & 2 \end{bmatrix},
\qquad
\hat{Y} = X \hat{W}^\top = \begin{bmatrix} 0 & 2.25 \\ 4.25 & 0.5 \\ -3.25 & 2.25 \\ 2.25 & 1.75 \end{bmatrix}.
$$

Six entries differ by $0.25$, so $\lVert Y - \hat{Y} \rVert_F^2 = 6 \cdot 0.0625 = 0.375$, while $\lVert Y \rVert_F^2 = 45.75$. Taking the square root of their ratio gives $\sqrt{0.375 / 45.75} = 0.0905$. Changing two weights by $0.25$ shifts the layer output by about 9%.

## Constraints

- `mse`, `sqnr_db`, and `max_abs_error` are compared with the reference using a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. An infinite SQNR must be exactly $+\infty$.
- `output_error` is compared using a relative tolerance of `1e-3`, because subtracting two nearly equal matrix products amplifies rounding error. In the tests, the true value is at least $0.005$, so computing $X (W - \hat{W})^\top$ instead of $Y - \hat{Y}$ produces the same result within the tolerance.
- Inputs have at most 100,000 elements, and all values are finite with magnitude at most $10^5$.
- No Python loops over elements are needed.

## Why this matters

Quantization trades numerical accuracy for memory and speed, and these four metrics guide that trade. Calibration routines pick clipping ranges by minimizing MSE, mixed-precision searches allocate extra bits to layers with the lowest SQNR, and release checks verify that no single weight moves beyond a bound using maximum absolute error.

Those first three metrics inspect weights in isolation, whereas `output_error` measures what the rest of the network actually receives. A weight error matters in proportion to the activations that multiply it, which is why modern LLM quantizers like GPTQ and AWQ minimize layer output error on calibration inputs rather than weight MSE. A metric with a silent bug, such as a wrong decibel factor or the wrong denominator, will never raise an exception. Instead, it misranks layers and makes a degraded model look healthy.
