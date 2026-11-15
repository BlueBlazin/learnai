## The idea

**The objective.** For one row $w$ of the weight and its quantized version $\hat w$,
the squared output error on the calibration inputs is a quadratic form in the weight
error $d = w - \hat w$:

$$
\lVert X w^\top - X \hat w^\top \rVert^2 = d\, X^\top X\, d^\top \propto \tfrac12\, d\, H\, d^\top,
\qquad H = \tfrac{2}{N} X^\top X .
$$

The rows do not interact and share one $H$, so the whole matrix can be processed
with the same small matrix.

**One step.** Suppose the weights in a set $F$ are still free and we fix one of
them, $w_q$, to its rounded value, an error of $\delta = w_q - \operatorname{quant}(w_q)$.
Minimizing the quadratic over the other free weights, subject to $w_q$ being fixed,
has a closed form (the Optimal Brain Surgeon update):

$$
\Delta w_F = -\frac{\delta}{[H_F^{-1}]_{qq}}\, \big(H_F^{-1}\big)_{q,:}\ ,
$$

where $H_F$ is the Hessian restricted to the free weights. The update moves the
other weights along row $q$ of the *inverse* Hessian of the free set.

**The catch, and the trick.** After fixing $w_q$, the free set shrinks, and the next
step needs $H_{F \setminus q}^{-1}$, which is not a submatrix of $H_F^{-1}$: removing
a variable from an inverse is a Gaussian elimination step. Doing it explicitly after
every column costs $O(\text{in}^2)$ each time and is numerically fragile. GPTQ's
observation: if the columns are quantized in a fixed order, $0, 1, 2, \dots$, then
all the rows that will ever be needed are known in advance, and they are the rows of
one Cholesky factor. With $H^{-1} = U^\top U$ and $U$ upper triangular, the inverse
Hessian of the columns $j, j+1, \dots$ (with $0..j-1$ eliminated) is

$$
H_{F_j}^{-1} = U_{j:,\,j:}^\top\, U_{j:,\,j:},
\qquad\text{whose first row is}\qquad U_{jj} \cdot U_{j,\,j:} .
$$

Substituting into the update, with $[H_{F_j}^{-1}]_{jj} = U_{jj}^2$:

$$
\Delta w_{j:} = -\frac{\delta}{U_{jj}^2}\, U_{jj}\, U_{j,\,j:} = -\frac{\delta}{U_{jj}}\, U_{j,\,j:} .
$$

That is the loop of the specification: divide the error by the diagonal entry
**once**, and move along row $j$ of the factor. The Cholesky factorization *is* the
sequence of eliminations, done once, stably.

**Dampening.** $X^\top X$ is singular when there are fewer samples than channels and
nearly singular whenever channels are strongly correlated. Adding
$\lambda = \texttt{percdamp} \cdot \operatorname{mean}(\operatorname{diag} H)$ to the
diagonal bounds the condition number (by about $\text{in} / \texttt{percdamp}$ in
the worst case) and, in terms of the objective, adds a penalty on the size of the
weight changes. It is scaled by the mean diagonal so that it means the same thing
for layers whose activations differ by orders of magnitude.

## Reference solution (PyTorch)

```python
def inverse_hessian_factor(x_calib, percdamp):
    n, in_dim = x_calib.shape
    h = (2.0 / n) * (x_calib.T @ x_calib)
    h = h + percdamp * torch.diagonal(h).mean() * torch.eye(in_dim)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(h_inv, upper=True)


def gptq_quantize(w, u, bits):
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    codes = []
    for j in range(w.shape[1]):
        q = torch.clamp(torch.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        err = (w[:, j] - q * scale) / u[j, j]
        w = w - torch.outer(err, u[j])
    return torch.stack(codes, dim=1).to(torch.int8), scale
```

The update is written functionally (`w = w - ...`), so the same lines work in JAX,
where arrays are immutable, and the caller's `w` is not modified. Because `u[j]` is
zero before column `j`, the full-width outer product only changes columns `j` and
later. MLX's `mx.linalg` factorizations run on the CPU and need `stream=mx.cpu`;
JAX has no `cholesky_inverse`, so its reference inverts the lower factor with a
triangular solve and multiplies.

**Numerics.** The factor is the only ill-conditioned step. In float32 its error is
roughly the condition number of the dampened Hessian times $10^{-7}$; with 1 percent
dampening that is far below what could change a code, which is why the published
implementation runs in float32 as well. Without dampening the first Cholesky can
fail outright, and that failure is the usual first symptom of a calibration set that
is too small.

## Common mistakes

- **Plain rounding.** Leaving out the update returns valid codes: round-to-nearest.
  Only a comparison of output errors notices.
- **Dividing by the wrong diagonal.** $U_{jj}^2$ is the diagonal of the inverse
  Hessian of the free set; paired with the Cholesky row, which already contains one
  factor of $U_{jj}$, the divisor is $U_{jj}$. Using $U_{00}$ for every column, or
  not dividing at all, are the same family.
- **Rows of $H^{-1}$ instead of rows of $U$.** $U^\top U$ is the inverse Hessian of
  *all* columns. Its row $j$ is the right update only for $j = 0$; afterwards it
  ignores that the earlier columns are fixed. This is the mistake the Cholesky
  formulation exists to prevent.
- **Column $j$ of the factor, or the lower factor.** Both are zero *after* index $j$
  and non-zero before it: the corrections go to columns that are already final, and
  the remaining ones get none.
- **The wrong sign.** Adding the correction doubles the damage.
- **No clamp.** A compensated weight can exceed the row's original maximum. Without
  the clamp its code leaves $[-q_{\max}, q_{\max}]$ and, at 8 bits, wraps in int8.
- **Recomputing the scale inside the loop.** The grid is fixed from the original
  weights. A scale that follows the working matrix differs per column, and the one
  scale that is stored no longer matches the codes.
- **No dampening, or absolute dampening.** `percdamp` is a fraction of the mean
  diagonal of $H$, not a number added as is.
- **The wrong factor.** The factor of $H$ instead of $H^{-1}$, the unfactored
  inverse, or $H$ without its $2/N$ all give a matrix of the right shape.

## Complexity

The factor costs $O(N \cdot \text{in}^2)$ for $X^\top X$ and $O(\text{in}^3)$ for the
factorizations, once per layer. The loop performs `in` outer-product updates of an
`(out, in)` matrix: $O(\text{out} \cdot \text{in}^2)$ in total, with no per-row
linear algebra because all rows share $U$. Memory is one `(in, in)` matrix beside
the weight.

## In the wild

**The paper and its code.** "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (arXiv 2210.17323, ICLR 2023), Algorithm 1. In
the authors' implementation (`IST-DASLab/gptq`, `gptq.py`), `add_batch` accumulates
$H$ as a running average of $\tfrac{2}{n} x x^\top$; `fasterquant` adds
`damp = percdamp * torch.mean(torch.diag(H))` to the diagonal (default
`percdamp = 0.01`), then computes `torch.linalg.cholesky(H)`,
`torch.cholesky_inverse(H)`, and `torch.linalg.cholesky(H, upper=True)`; the column
loop is `err1 = (w - q) / d` with `d = Hinv1[i, i]`, followed by
`W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))`. The quantizer's
parameters are found once, before the loop, from the original weights. This problem
follows those steps. `mlx-lm` 0.32 has the same algorithm in `mlx_lm/quant/gptq.py`,
with the factorizations inside `mx.stream(mx.cpu)`.

**What is left out.**

- *Lazy batching.* The reference processes columns in blocks of 128: inside a block
  it updates only the block's columns, and it applies the accumulated errors to all
  later columns once per block (`W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])`). The
  result is the same; the memory traffic is far lower. That is the "without lazy
  batching" in this problem's scope.
- *Dead channels.* A channel that is always zero has a zero on the diagonal of $H$;
  the reference sets that diagonal entry to 1 and the channel's weights to 0. Here
  every channel is active.
- *The grid.* The reference uses an asymmetric min-max grid, optionally per group
  of columns and optionally recomputed per group during the loop, and an option to
  process columns in order of decreasing Hessian diagonal (`actorder`). This problem
  uses the track's symmetric per-row grid and the natural order.

**GPTQ and AWQ.** Both spend the same calibration activations. AWQ rescales channels
so that plain rounding hurts less; GPTQ keeps the weights where they are and makes
the rounding of each column depend on the errors already made. GPTQ is the more
aggressive of the two and can overfit a small calibration set, since it minimizes
the error on exactly those samples; the dampening is its only regularizer.
