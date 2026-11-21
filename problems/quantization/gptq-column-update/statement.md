Implement the core of GPTQ: quantize a weight matrix one column at a time, and after
each column, adjust the columns not yet quantized so that they compensate the error
just made, as measured on calibration inputs. The compensation needs the inverse of
the layer's Hessian with the already quantized columns eliminated, and the trick of
the method is that every such inverse is one row of a single Cholesky factor. You
build that factor, then run the column loop. This is post-training calibration: no
gradients, no optimizer.

## Specification

```python
def inverse_hessian_factor(x_calib, percdamp): ...  # -> u
def gptq_quantize(w, u, bits): ...  # -> (codes, scale)
```

`w` is the weight of a linear layer, shape `(out, in)`, mapping an input row $x$ to
$x W^\top$. `x_calib` holds $N$ calibration inputs, shape `(N, in)`.

**`inverse_hessian_factor(x_calib, percdamp)`.** With $X$ = `x_calib`:

1. **Hessian.** $H = \frac{2}{N} X^\top X$, shape `(in, in)`. It is the Hessian of
   the squared output error $\lVert X w^\top - X \hat w^\top \rVert^2 / N$ with
   respect to one row $\hat w$ of the weight, the same for every row.
2. **Dampening.** $H \leftarrow H + \lambda I$ with
   $\lambda = \texttt{percdamp} \cdot \operatorname{mean}(\operatorname{diag}(H))$.
3. **Factor.** Return the upper-triangular matrix $U$ with positive diagonal such
   that
   $$ H^{-1} = U^\top U . $$
   This is the transpose of the lower Cholesky factor of $H^{-1}$.

**`gptq_quantize(w, u, bits)`.** The grid is the symmetric per-row grid of
`symmetric-per-channel`, fixed **before** the loop from the original weights. With
$q_{\max} = 2^{\text{bits}-1} - 1$:

$$
m_i = \max_j |w_{ij}|, \qquad
\text{scale}_i = \begin{cases} m_i / q_{\max} & m_i > 0 \\ 1 & m_i = 0 . \end{cases}
$$

Then, with $W$ a working copy of `w`, for $j = 0, 1, \dots, \text{in} - 1$ in order:

$$
\begin{aligned}
q_{ij} &= \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{W_{ij}}{\text{scale}_i}\Big),\ -q_{\max},\ q_{\max}\Big)
&& \text{(quantize column } j \text{ as it is now)} \\[0.5ex]
e_i &= \frac{W_{ij} - q_{ij} \cdot \text{scale}_i}{U_{jj}}
&& \text{(its error, scaled)} \\[0.5ex]
W_{ik} &\leftarrow W_{ik} - e_i \, U_{jk} \quad \text{for all } k > j
&& \text{(compensate in the remaining columns)}
\end{aligned}
$$

Every row $i$ is processed independently and identically. Columns before $j$ are
final and are never touched again; the clamp applies because a compensated weight
can leave the row's original range.

Return `(codes, scale)`: `codes` is the **int8** tensor of the $q_{ij}$, shape
`(out, in)`; `scale` is float32 of shape `(out,)`. The quantized weight is
$\hat W_{ij} = q_{ij} \cdot \text{scale}_i$.

- `x_calib`: float32, shape `(N, in)`, $N \ge 1$. $N$ may be smaller than `in`.
  Every input channel is non-zero in at least one sample.
- `percdamp`: a Python float $\ge 0$. It is positive whenever $X^\top X$ is singular
  or nearly so.
- `u`: float32, shape `(in, in)`, upper triangular with a positive diagonal. In the
  tests of `gptq_quantize` it is computed by the judge; it is the factor of a
  dampened inverse Hessian, or, in the worked example, a small matrix with round
  numbers.
- `bits`: a Python int, $2 \le \text{bits} \le 8$.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer, ties to
even, as in PyTorch, JAX, and MLX.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x_calib` | `(N, in)` | float32 |
| `u` | `(in, in)` | float32 |
| `w` | `(out, in)` | float32 |
| `codes` | `(out, in)` | int8 |
| `scale` | `(out,)` | float32 |

## Example

```python
w = [[0.7, 1.5, -0.8], [1.4, -3.0, 0.4]]
u = [[1.0, 0.5, -0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]

codes, scale = gptq_quantize(w, u, bits=3)
# scale = [0.5, 1.0]
# codes = [[1,  3, -1],
#          [1, -3,  1]]
```

With 3 bits, $q_{\max} = 3$, so the scales are $1.5 / 3 = 0.5$ and $3.0 / 3 = 1$.
Row 0, column by column:

| $j$ | $W_{0j}$ now | $W_{0j} / \text{scale}$ | $q$ | $q \cdot \text{scale}$ | $e = \text{error} / U_{jj}$ | row 0 of $W$ after the update |
|---|---|---|---|---|---|---|
| 0 | $0.7$ | $1.4$ | $1$ | $0.5$ | $0.2$ | $[\,\cdot\,,\ 1.5 - 0.2 \cdot 0.5,\ -0.8 + 0.2 \cdot 0.5] = [\,\cdot\,,\ 1.4,\ -0.7]$ |
| 1 | $1.4$ | $2.8$ | $3$ | $1.5$ | $-0.1$ | $[\,\cdot\,,\ \cdot\,,\ -0.7 + 0.1 \cdot 0.5] = [\,\cdot\,,\ \cdot\,,\ -0.65]$ |
| 2 | $-0.65$ | $-1.3$ | $-1$ | $-0.5$ | | |

![Row 0 of the example, quantized one column at a time. The error of each column is added to the columns after it.](assets/column-steps.svg)

Rounding each weight on its own would give $[1, 3, -2]$: the last weight, $-0.8$,
is $1.6$ steps from zero. GPTQ rounds it to $-1$ instead, because the first two
columns were rounded in directions whose effect on the output is best cancelled by
a less negative third weight. Row 1 works the same way and ends with code $1$ where
plain rounding gives $0$.

Measured with the Hessian that this factor belongs to, $H = (U^\top U)^{-1}$, the
output error $(w - \hat w) H (w - \hat w)^\top$ of the two rows is $0.0725$ and
$0.29$ for these codes, against $0.1725$ and $0.69$ for plain rounding.

## Constraints

- $1 \le \text{out} \le 64$, $2 \le \text{in} \le 64$, $1 \le N \le 256$. Entries
  of `w` and `x_calib` are finite with magnitude at most $10^3$.
- **`gptq_quantize`**: `codes` are compared **exactly** and must be int8; `scale` to
  a relative tolerance of `1e-4`. In float32 a compensated weight carries an error
  of about $10^{-4}$ of a step after 64 columns, so in the exact-comparison tests no
  weight reaches its own column within $0.02$ of a rounding tie.
- **`inverse_hessian_factor`** is compared with a float64 computation, entry by
  entry, to
  $|\text{got} - \text{expected}| \le 10^{-3}\,|\text{expected}| + 10^{-4} \max|U|$.
  The dampened Hessians of the tests have condition numbers up to about $10^{3}$,
  for which a float32 Cholesky factorization of the inverse is accurate to about
  $5 \cdot 10^{-6} \max|U|$ in all three frameworks. Compute in float32; float64 is
  not available on the MLX GPU and is not needed.
- **Property.** On correlated calibration inputs, the output error
  $\lVert X (W - \hat W)^\top \rVert^2$ of your codes must be at most $0.9$ times
  that of round-to-nearest on the same grid, both with the judge's factor and end to
  end with your own. The reference reaches $0.1$ to $0.7$.
- A Python loop over the columns is expected. No loops over rows are needed. Your
  framework's Cholesky factorization and triangular or Cholesky inverses are allowed
  (in MLX they run on the CPU: pass `stream=mx.cpu`).

## Why this matters

Round-to-nearest treats every weight as if it were alone. It is not: the layer's
output is a sum over weights whose inputs are correlated, so an error on one weight
can be cancelled by a deliberate, opposite error on its neighbors. GPTQ does exactly
that, one column at a time, and it was the method that first made 3- and 4-bit
quantization of large language models work in a few GPU hours without retraining;
its descendants are behind the calibration-based quantizers of llm-compressor and
`mlx-lm`. The algorithm is a dozen lines, and its failure modes are quiet. Dividing
by the wrong diagonal entry, using rows of the inverse Hessian instead of its
Cholesky factor, or skipping the dampening on a rank-deficient calibration set all
still return valid codes. The model then loads, generates fluent text, and is worse
than plain rounding would have made it.
