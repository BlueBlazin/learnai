## The idea

All four metrics are norms of the error $e = a - b$, normalized in different ways.

| metric | norm of the error | normalized by | unit |
|---|---|---|---|
| `mse` | $\lVert e \rVert_2^2$ | the number of elements | units of the tensor, squared |
| `sqnr_db` | $\lVert e \rVert_2^2$ | $\lVert \text{ref} \rVert_2^2$, inverted, on a log scale | dB |
| `max_abs_error` | $\lVert e \rVert_\infty$ | nothing | units of the tensor |
| `output_error` | $\lVert X e^\top \rVert_F$ | $\lVert X W^\top \rVert_F$ | dimensionless |

**MSE** is the average noise power. It depends on the scale of the tensor: the same
8-bit quantizer gives an MSE ten thousand times smaller on a layer whose weights are a
hundred times smaller. It is the right objective *within* one tensor (choosing a
clipping range, comparing rounding schemes) and meaningless across tensors.

**SQNR** fixes that by dividing the signal power by the noise power, and the logarithm
turns the huge range of that ratio into small numbers. Decibels are defined for power
ratios as $10 \log_{10}$. For amplitudes $A$, power goes as $A^2$, so the same quantity
is $10 \log_{10} (A_1^2 / A_2^2) = 20 \log_{10} (A_1 / A_2)$: the 20 appears only when
the ratio is one of amplitudes. Sums of squares are powers, so the factor is 10. The
$1/N$ of the two means cancels, which is why sums suffice. A useful anchor: each extra
bit of a uniform quantizer halves the step, quarters the noise power, and adds
$10 \log_{10} 4 \approx 6.02$ dB.

**Maximum absolute error** is the worst case. For a uniform quantizer without
clipping it is at most half a step, so it is a direct test of an implementation: a
value above half a step means clipping or a bug.

**Output error** measures the layer, not the weights. By linearity
$Y - \hat{Y} = X (W - \hat{W})^\top$: the weight error, seen through the inputs. Two
approximations with the same weight MSE can have very different output errors,
depending on whether the error falls on input directions that carry large activations.

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

Reductions without an axis argument run over every element and return a 0-d tensor.
When the noise is zero, `signal / noise` is `inf` and its logarithm is `inf`: IEEE
arithmetic produces the specified answer with no special case. The JAX and MLX
versions are word-for-word translations.

`output_error` could equally compute `x @ (w - w_hat).T`, which is one product instead
of two and more accurate, since it does not subtract two nearly equal matrices. With
float32 and errors of a percent or more the two agree to about five digits.

## Common mistakes

- **`20 log10` on a power ratio.** Every SQNR doubles: 20 dB is reported as 40 dB.
  The result looks reasonable, which is why this survives in real codebases; the check
  is that a 10% error must give 20 dB.
- **Normalizing by the approximation.** Dividing by $\lVert \hat{Y} \rVert$ instead
  of $\lVert Y \rVert$ (or taking the signal power from `approx`). The difference is
  small when the approximation is good and grows exactly when the metric matters:
  an approximation that shrinks everything looks worse than it is, one that inflates
  everything looks better.
- **Reducing over one axis.** `mean(dim=-1)` returns a vector of per-row errors. It
  is right for a 1-D input, which is what a first test usually is.
- **A mean of per-row ratios.** Computing the relative error of each output row and
  averaging gives every input the same weight, including inputs whose output is
  nearly zero, where the ratio is huge and meaningless. The ratio of the norms
  weights rows by their energy.
- **`x @ w` instead of `x @ w.T`.** The weight is stored `(out, in)`. Without the
  transpose the shapes only fit a square layer, and then the product is the output of
  a different layer.
- **Forgetting the absolute value** in `max_abs_error`. If the approximation is above
  the reference everywhere, `max(a - b)` is negative.
- **An epsilon in the denominator.** `signal / (noise + eps)` replaces the honest
  $+\infty$ of an exact approximation by a number that depends on `eps`, and caps
  every SQNR that can be reported.

## Complexity

The elementwise metrics are $O(N)$ time and memory for the difference. `output_error`
costs two matrix products, $O(n \cdot \text{in} \cdot \text{out})$.

## In the wild

SQNR in decibels is the standard per-layer diagnostic in quantization toolkits; Nagel
et al. (2021), "A White Paper on Neural Network Quantization", use it to find the
layers responsible for an accuracy drop. Range setting by minimizing the MSE between
a tensor and its quantized version is the default calibration in most of them.

The output error is the objective of the weight-quantization methods used for LLMs.
GPTQ (Frantar et al., 2022) minimizes $\lVert X W^\top - X \hat{W}^\top \rVert_F^2$
layer by layer, quantizing one column at a time and compensating in the remaining
ones; AWQ (Lin et al., 2023) searches per-channel scalings that minimize the same
quantity. Both need only a few hundred calibration rows `x`, because what matters is
the second-moment matrix $X^\top X$: indeed
$\lVert X E^\top \rVert_F^2 = \operatorname{tr}(E\, X^\top X\, E^\top)$ with
$E = W - \hat{W}$, so the calibration set can be reduced to an `in` by `in` matrix once
and reused for every candidate.

None of these numbers is the final word. They are proxies, cheap enough to compute
inside a search loop; the metric that decides whether a quantized model ships is
measured on its outputs, as perplexity or the KL divergence of its token distributions
against the original.
