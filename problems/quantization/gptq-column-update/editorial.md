## The idea

**The objective.** For a single weight row $w$ and its quantized version $\hat w$, the squared output error on the calibration inputs $X$ is a quadratic form in the weight error $d = w - \hat w$:

$$
\lVert X w^\top - X \hat w^\top \rVert^2 = d\, X^\top X\, d^\top \propto \tfrac12\, d\, H\, d^\top,
\qquad H = \tfrac{2}{N} X^\top X .
$$

Because the rows do not interact and all share the same Hessian $H$, one small `(in, in)` matrix governs the entire layer.

**One column step.** Let $F$ be the set of columns not yet quantized. When we quantize one weight $w_q$ in $F$, we introduce an error $\delta = w_q - \operatorname{quant}(w_q)$. Minimizing the quadratic output error over the remaining free weights while holding $w_q$ fixed gives the Optimal Brain Surgeon update:

$$
\Delta w_F = -\frac{\delta}{[H_F^{-1}]_{qq}}\, \big(H_F^{-1}\big)_{q,:}\ ,
$$

where $H_F$ is the Hessian restricted to the free columns $F$. The update shifts the remaining weights along row $q$ of the inverse Hessian of the free set.

**The Cholesky trick.** After fixing column $q$, the free set shrinks to $F \setminus \{q\}$. The next step needs $H_{F \setminus \{q\}}^{-1}$, which is not simply a submatrix of $H_F^{-1}$ because eliminating a variable from an inverse requires a rank-1 Gaussian elimination update. Doing that explicitly after every column costs $O(\text{in}^2)$ per step and accumulates numerical error.

GPTQ exploits the fact that all columns are quantized in the fixed order $0, 1, 2, \dots$. Because the elimination order is known in advance, every required row can be precomputed in a single Cholesky factorization. Writing $H^{-1} = U^\top U$ with $U$ upper triangular, the inverse Hessian of the remaining columns $F_j = \{j, j+1, \dots, \text{in}-1\}$ (with columns $0, \dots, j-1$ already eliminated) is:

$$
H_{F_j}^{-1} = U_{j:,\,j:}^\top\, U_{j:,\,j:},
\qquad\text{whose first row is}\qquad U_{jj} \cdot U_{j,\,j:} .
$$

Its top-left diagonal entry is $[H_{F_j}^{-1}]_{jj} = U_{jj}^2$. Substituting both into the update cancels one factor of $U_{jj}$:

$$
\Delta w_{j:} = -\frac{\delta}{U_{jj}^2}\, U_{jj}\, U_{j,\,j:} = -\frac{\delta}{U_{jj}}\, U_{j,\,j:} .
$$

This gives the loop in the specification. You divide the error by $U_{jj}$ once and subtract that scaled error times row $j$ of $U$. The Cholesky factorization performs the entire sequence of Gaussian eliminations in one numerically stable step.

**Dampening.** Whenever $N < \text{in}$, $X^\top X$ is singular, and strongly correlated channels make it nearly singular even for large $N$. Adding $\lambda = \texttt{percdamp} \cdot \operatorname{mean}(\operatorname{diag} H)$ to the diagonal bounds the condition number (roughly by $\text{in} / \texttt{percdamp}$ in the worst case) and acts as an $L_2$ penalty on the weight updates. Scaling $\lambda$ by the mean diagonal keeps `percdamp` invariant to the activation scale of each layer.

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

Writing the update functionally (`w = w - ...`) avoids mutating the caller's tensor and works directly in JAX, where arrays are immutable. Because `u[j]` is zero before column `j`, the full-width outer product leaves earlier columns untouched. In MLX, `mx.linalg` routines run on CPU and require `stream=mx.cpu`. JAX has no `cholesky_inverse`, so its reference inverts the lower Cholesky factor with a triangular solve and multiplies by its transpose.

**Numerics.** Computing `u` is the only ill-conditioned step. In float32, its error is roughly the dampened Hessian's condition number times $10^{-7}$. With 1 percent dampening, that error is far too small to flip quantization codes, which is why the published GPTQ implementation also runs in float32. Without dampening, the first Cholesky factorization often fails on small calibration sets.

## Common mistakes

- **Plain rounding.** Skipping the column update still returns valid round-to-nearest codes, and only the output-error check catches it.
- **Dividing by the wrong diagonal.** Since $[H_{F_j}^{-1}]_{jj} = U_{jj}^2$ and row $j$ of $U$ already cancels one factor of $U_{jj}$, you must divide by $U_{jj}$. Dividing by $U_{jj}^2$, dividing by $U_{00}$ for every column, or omitting the division all break the update scale.
- **Using rows of $H^{-1}$ instead of rows of $U$.** $U^\top U$ is the inverse Hessian of all columns. Its row $j$ is only valid at $j = 0$, because later steps must account for earlier columns already being fixed.
- **Using column $j$ of $U$ or the lower Cholesky factor.** Both are zero after index $j$ and non-zero before it, so the update modifies already-finalized columns and leaves the remaining columns unchanged.
- **Wrong sign on the update.** Adding `outer(err, u[j])` instead of subtracting it doubles the output error.
- **Missing clamp.** Compensated weights can exceed the row's original maximum. Without clamping to $[-q_{\max}, q_{\max}]$, codes overflow the target bit width and wrap in `int8` at 8 bits.
- **Recomputing `scale` inside the loop.** The grid is fixed from the original `w`. Recomputing `scale` from the evolving working matrix makes each column use a different grid that no longer matches the returned `scale`.
- **Missing or absolute dampening.** `percdamp` is a fraction of $\operatorname{mean}(\operatorname{diag} H)$, not a raw constant added directly to the diagonal.
- **Returning the wrong matrix.** Factoring $H$ instead of $H^{-1}$, returning $H^{-1}$ unfactored, or dropping the $2/N$ factor all produce a matrix of the right shape with wrong values.

## Complexity

Building the factor takes $O(N \cdot \text{in}^2)$ time for $X^\top X$ and $O(\text{in}^3)$ for the Cholesky operations, once per layer. The column loop performs `in` outer-product updates on an `(out, in)` matrix for $O(\text{out} \cdot \text{in}^2)$ total time, with no per-row linear algebra because all rows share $U$. Extra memory is one `(in, in)` matrix alongside the weights.

## In the wild

**The paper and reference code.** This problem follows Algorithm 1 of "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (arXiv 2210.17323, ICLR 2023). In `IST-DASLab/gptq` (`gptq.py`), `add_batch` accumulates $H$ as a running average of $\frac{2}{n} x x^\top$. Then `fasterquant` adds `damp = percdamp * torch.mean(torch.diag(H))` to the diagonal (default `percdamp = 0.01`), runs `torch.linalg.cholesky(H)`, `torch.cholesky_inverse(H)`, and `torch.linalg.cholesky(H, upper=True)`, and executes the column loop `err1 = (w - q) / d` with `d = Hinv1[i, i]` followed by `W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))`. The quantization parameters are computed once before the loop from the original weights. `mlx-lm` 0.32 implements the same algorithm in `mlx_lm/quant/gptq.py` inside `mx.stream(mx.cpu)`.

**What this problem omits:**

- *Lazy batching.* The reference processes columns in blocks of 128, updating only within the current block per column and applying accumulated errors to all later columns once per block (`W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])`). The mathematical result is identical, with much lower memory bandwidth.
- *Dead channels.* An always-zero input channel produces a zero diagonal entry in $H$. The reference sets that diagonal entry to $1$ and zeroes the corresponding weight column. Here every channel is active.
- *Grid variants and column ordering.* The reference supports asymmetric min-max grids, group-wise scales (optionally recomputed per group), and `actorder`, which sorts columns by decreasing Hessian diagonal. This problem uses the track's symmetric per-row grid and the natural column order.

**GPTQ vs. AWQ.** Both methods use calibration activations, but they act differently. AWQ rescales channels before quantization so that plain rounding causes less damage, whereas GPTQ leaves the original scaling intact and adapts each column's rounding to cancel earlier errors. GPTQ is more aggressive and can overfit a small calibration set because it directly minimizes error on those samples, with dampening as its only regularizer.
