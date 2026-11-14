Implement the four numbers that are used to say how much a quantized tensor differs
from the original: mean squared error, signal-to-quantization-noise ratio in decibels,
maximum absolute error, and the relative error of a linear layer's output. Each is one
line; each has a classic way to be wrong by a constant factor or an axis that still
produces a plausible number.

## Specification

```python
def mse(a, b): ...
def sqnr_db(ref, approx): ...
def max_abs_error(a, b): ...
def output_error(w, w_hat, x): ...
```

Every function returns a **0-d float32 tensor**: one number for the whole input,
whatever its shape. Below, sums, means, and maxima run over all $N$ elements of a
tensor.

**`mse(a, b)`**, the mean squared error:

$$
\operatorname{mse}(a, b) = \frac{1}{N} \sum_{k} (a_k - b_k)^2 .
$$

**`sqnr_db(ref, approx)`**, the signal-to-quantization-noise ratio in decibels. The
signal power is that of the reference, and the noise is the difference:

$$
\operatorname{sqnr\_db}(\text{ref}, \text{approx}) =
10 \log_{10} \frac{\sum_k \text{ref}_k^2}{\sum_k (\text{ref}_k - \text{approx}_k)^2} .
$$

The argument order matters: the numerator uses `ref` only. If `approx` equals `ref` in
every element, the noise is 0 and the result is $+\infty$; return that, not a large
finite number and not NaN. `ref` is never all zeros.

**`max_abs_error(a, b)`**, the worst single element:

$$
\operatorname{max\_abs\_error}(a, b) = \max_k\, |a_k - b_k| .
$$

**`output_error(w, w_hat, x)`**, the relative error at the output of a linear layer.
`w` is the layer's weight of shape `(out, in)`, `w_hat` its approximation of the same
shape, and `x` a batch of `n` input rows of shape `(n, in)`. The layer computes
$Y = X W^\top$, of shape `(n, out)`, and the approximate layer $\hat{Y} = X
\hat{W}^\top$. The result is

$$
\operatorname{output\_error}(W, \hat{W}, X) =
\frac{\lVert Y - \hat{Y} \rVert_F}{\lVert Y \rVert_F},
\qquad \lVert A \rVert_F = \sqrt{\sum_{i,j} A_{ij}^2} ,
$$

where both Frobenius norms run over all $n \cdot \text{out}$ entries. The denominator
is the norm of the **exact** output. $Y$ is never all zeros.

Write the arithmetic yourself: do not call a built-in mean-squared-error loss
(`mse_loss`). Elementwise operations, reductions (`sum`, `mean`, `max`), `log10`,
`sqrt`, and matrix multiplication are what you need.

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

The differences are $[-0.1, 0, 0.2, 0]$, with squares $[0.01, 0, 0.04, 0]$. Their mean
is $0.05 / 4 = 0.0125$ and the largest magnitude is $0.2$. The signal power is
$1 + 4 + 9 + 0.25 = 14.25$ and the noise power $0.05$, so the ratio is $285$ and the
SQNR is $10 \log_{10} 285 = 24.55$ dB.

```python
w = [[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]]  # (out, in) = (2, 3)
w_hat = [[1.0, 2.25, -1.0], [0.5, -0.5, 1.75]]
x = [[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0]]  # n = 4

output_error(w, w_hat, x)  # 0.0905357
```

$$
Y = X W^\top = \begin{bmatrix} 0 & 2.5 \\ 4 & 0.5 \\ -3 & 2.5 \\ 2 & 2 \end{bmatrix},
\qquad
\hat{Y} = X \hat{W}^\top = \begin{bmatrix} 0 & 2.25 \\ 4.25 & 0.5 \\ -3.25 & 2.25 \\ 2.25 & 1.75 \end{bmatrix}.
$$

Six entries differ by $0.25$, so $\lVert Y - \hat{Y} \rVert_F^2 = 6 \cdot 0.0625 =
0.375$, and $\lVert Y \rVert_F^2 = 45.75$. The result is $\sqrt{0.375 / 45.75} =
0.0905$: the two weights that changed by $0.25$ move the output by 9%.

## Constraints

- `mse`, `sqnr_db`, and `max_abs_error` are compared with the reference to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-6`. An infinite SQNR must be
  exactly $+\infty$.
- `output_error` is compared to a relative tolerance of `1e-3`, because it subtracts
  two matrix products that are nearly equal. In the tests the true value is at least
  $0.005$, so computing $X (W - \hat{W})^\top$ instead of $Y - \hat{Y}$ gives the same
  result within the tolerance.
- Inputs have at most 100,000 elements, and all values are finite with magnitude at
  most $10^5$.
- No Python loops over elements are needed.

## Why this matters

Quantization is a trade of accuracy for memory, and these are the numbers the trade is
made with. A calibration method chooses a clipping range by minimizing the MSE; a
mixed-precision search gives more bits to the layers whose SQNR is lowest; a release
check asserts that no weight moved by more than a bound. The first three measure the
weights. The fourth measures what the model sees: an error in a weight matters in
proportion to the activations that multiply it, which is why modern methods such as
GPTQ and AWQ minimize the output error on calibration inputs and not the weight error.
A metric that is off by a factor of two in decibels, or normalized by the wrong
tensor, does not crash. It ranks layers in the wrong order and reports a model as
better than it is.
