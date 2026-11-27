## The idea

All four metrics measure a norm of the error $e = a - b$, normalized in different ways.

| metric | norm of the error | normalized by | unit |
|---|---|---|---|
| `mse` | $\lVert e \rVert_2^2$ | the number of elements | units of the tensor, squared |
| `sqnr_db` | $\lVert e \rVert_2^2$ | $\lVert \text{ref} \rVert_2^2$, inverted, on a log scale | dB |
| `max_abs_error` | $\lVert e \rVert_\infty$ | nothing | units of the tensor |
| `output_error` | $\lVert X e^\top \rVert_F$ | $\lVert X W^\top \rVert_F$ | dimensionless |

**MSE** is the average noise power. Because it scales with the square of the tensor's magnitude, the same 8-bit quantizer produces an MSE 10,000 times smaller on weights that are 100 times smaller. MSE is therefore useful *within* a single tensor (for picking a clipping range or comparing rounding schemes), but it cannot be compared across different layers.

**SQNR** removes that scale dependence by dividing signal power by noise power, and the logarithm compresses the wide range of ratios into readable numbers. Decibels for a power ratio use $10 \log_{10}$. Because power scales as amplitude squared ($A^2$), an amplitude ratio uses $10 \log_{10} (A_1^2 / A_2^2) = 20 \log_{10} (A_1 / A_2)$. Here both numerator and denominator are sums of squares (powers), so the factor is 10. The $1/N$ factor from taking means cancels out, so plain sums suffice. As a rule of thumb, each extra bit in a uniform quantizer halves the step size, quarters the noise power, and adds $10 \log_{10} 4 \approx 6.02$ dB.

**Maximum absolute error** captures the worst single element. For a uniform quantizer without clipping, it is bounded by half a step size, making it a direct sanity check. Any value larger than half a step points to clipping or an implementation bug.

**Output error** evaluates the layer rather than its weights alone. By linearity, $Y - \hat{Y} = X (W - \hat{W})^\top$, which projects the weight error through the input activations. Two quantized weights with identical weight MSE can produce very different output errors depending on whether the error aligns with high-magnitude input directions.

## Reference solution (PyTorch)

```python
def mse(a, b):
    return ((a - b) ** 2).mean()


def sqnr_db(ref, approx):
    signal = (ref**2).sum()
    noise = ((ref - approx) ** 2).sum()
    return 10.0 * torch.log10(signal / noise)


def max_abs_error(a, b):
    return (a - b).abs().max()


def output_error(w, w_hat, x):
    y = x @ w.T
    y_hat = x @ w_hat.T
    return torch.sqrt(((y - y_hat) ** 2).sum()) / torch.sqrt((y**2).sum())
```

Calling a reduction without an axis argument reduces over every element and returns a 0-d tensor. When `noise` is zero, IEEE floating-point arithmetic evaluates `signal / 0.0` to `inf` and `log10(inf)` to `inf`, producing $+\infty$ with no special case. The JAX and MLX implementations are direct translations.

In `output_error`, computing `x @ (w - w_hat).T` directly instead of `y - y_hat` saves a matrix multiplication and avoids subtracting two nearly equal matrices. In float32, when the relative error is at least a percent, both formulations agree to about five significant digits.

## Common mistakes

- **Using `20 * log10` on a power ratio.** Every SQNR value doubles, turning 20 dB into 40 dB. Because 40 dB still looks plausible, this bug often goes unnoticed. As a quick check, a uniform 10% error must give 20 dB.
- **Normalizing by the approximation.** Dividing by $\lVert \hat{Y} \rVert_F$ instead of $\lVert Y \rVert_F$, or using `approx` for the signal power in `sqnr_db`, biases the metric as error grows. An approximation that shrinks every entry looks worse than it is, while one that inflates every entry looks better.
- **Reducing over a single axis.** Calling `mean(dim=-1)` returns a vector of per-row errors instead of a scalar. This passes on 1-D inputs, which are often the first test cases people write.
- **Averaging per-row relative errors.** Computing the relative error of each row in `Y` and averaging gives every input equal weight. Inputs whose exact output is near zero produce huge, noisy ratios that dominate the average, whereas the ratio of Frobenius norms weights rows by their energy.
- **Writing `x @ w` instead of `x @ w.T`.** Weights have shape `(out, in)`. Omitting the transpose still runs on square weight matrices, silently computing the output of the transposed layer.
- **Forgetting the absolute value in `max_abs_error`.** If `approx` is greater than `ref` at every element, `(a - b).max()` returns a negative number.
- **Adding an epsilon to the denominator.** Writing `signal / (noise + eps)` replaces $+\infty$ on an exact match with an arbitrary constant and caps the maximum reportable SQNR.

## Complexity

The three elementwise metrics take $O(N)$ time and $O(N)$ memory to form the difference tensor. `output_error` performs two matrix multiplications and takes $O(n \cdot \text{in} \cdot \text{out})$ time.

## In the wild

SQNR in decibels is a standard per-layer diagnostic across quantization toolkits. Nagel et al. (2021), "A White Paper on Neural Network Quantization", use it to pinpoint which layers drive an accuracy drop. Minimizing weight MSE is also the default calibration objective for choosing clipping ranges in most toolkits.

Relative output error is the core objective behind modern LLM weight-quantization methods. GPTQ (Frantar et al., 2022) minimizes $\lVert X W^\top - X \hat{W}^\top \rVert_F^2$ layer by layer, quantizing one column at a time and updating the remaining columns to compensate. AWQ (Lin et al., 2023) searches for per-channel scales that minimize the same objective. Both methods need only a few hundred calibration rows `x` because the objective depends only on the second-moment matrix $X^\top X$. Writing $E = W - \hat{W}$, we have $\lVert X E^\top \rVert_F^2 = \operatorname{tr}(E\, X^\top X\, E^\top)$, so the calibration inputs can be collapsed once into an `in` by `in` matrix and reused for every candidate.

None of these metrics is the final measure of model quality. They are cheap proxies used inside optimization and search loops, while the final decision to ship a quantized model relies on end-to-end metrics like perplexity or the KL divergence between quantized and reference token distributions.
