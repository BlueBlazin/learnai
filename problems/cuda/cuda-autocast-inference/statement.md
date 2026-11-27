Run a model's forward pass in mixed precision, then compute log-probabilities and entropy from its output logits in float32. The network body runs faster and remains accurate in 16-bit floating point, but temperature scaling, log-softmax, and entropy sums do not. Once the model returns 16-bit logits, PyTorch will keep all subsequent operations in 16 bits unless you explicitly cast back to float32.

## Specification

```python
def forward_amp(model, x, device, dtype, temperature=1.0): ...  # returns (log_probs, entropy)
```

- `model`: a PyTorch module whose parameters are float32 and already on `device`. Given input `x` of batch size $B$ and feature dimension $D$, `model(x)` returns logits of shape `(B, V)`, where $V$ is the vocabulary size.
- `x`: input tensor on `device`.
- `device`: the `torch.device` of `model` and `x`, either `torch.device("cuda")` or `torch.device("cpu")`.
- `dtype`: the 16-bit floating-point type for the autocast region, either `torch.float16` or `torch.bfloat16`.
- `temperature`: a positive Python float $\tau$ used to scale the logits.

The function has two stages:

1. **Mixed-precision forward pass.** Call `model(x)` exactly once inside a `torch.autocast` region matching the device type of `device` and using `dtype`. Inside this region, PyTorch runs matrix multiplications and linear layers in `dtype` while leaving the model's stored parameters in float32. Let $z$ be the resulting 16-bit logits tensor of shape `(B, V)`.
2. **Float32 statistics.** Exit the autocast region, convert $z$ to float32, and divide by $\tau$ in float32. Then compute the log-probabilities $\ell$ (shape `(B, V)`) and per-example entropy $H$ (shape `(B,)`) in float32:

$$
\ell_{b,i} = \frac{z_{b,i}}{\tau} - \log \sum_{j=0}^{V-1} e^{z_{b,j} / \tau},
\qquad
H_b = -\sum_{i=0}^{V-1} e^{\ell_{b,i}}\, \ell_{b,i}.
$$

Return the tuple `(log_probs, entropy)` with both tensors in float32 on `device`.

Do not modify `model`. Its parameters must remain float32 after the call. Gradients are neither required nor checked.

**What the judge observes.** On each call to `model(x)`, the test harness records whether autocast was active for `device`, which 16-bit `dtype` was active, and the exact logits `model(x)` returned. Reference values for `log_probs` and `entropy` are computed in float64 from those exact logits, so your only error comes from float32 rounding.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, D)` | float32 |
| `model(x)` inside the region | `(B, V)` | `dtype` (`float16` or `bfloat16`) |
| `log_probs` | `(B, V)` | float32 |
| `entropy` | `(B,)` | float32 |

## Example

Consider a single linear layer without bias whose weight rows are $[40, 0]$, $[0, 40]$, and $[40, 40]$, with input `x = [[1.5, 0.25]]`. The raw logits are $[60, 10, 70]$, which both float16 and bfloat16 represent exactly.

```python
forward_amp(model, x, torch.device("cpu"), torch.bfloat16, 20.0)
# log_probs = [[-1.0046, -3.5046, -0.5046]]
# entropy   = [0.7779]
```

Dividing by $\tau = 20.0$ gives scaled logits $[3, 0.5, 3.5]$ and log-normalizer $\log(e^{3} + e^{0.5} + e^{3.5}) = \log 54.850 = 4.0046$. The corresponding probabilities are $[0.3662, 0.0301, 0.6038]$, giving $H = 0.3662 \cdot 1.0046 + 0.0301 \cdot 3.5046 + 0.6038 \cdot 0.5046 = 0.7779$.

```python
forward_amp(model, x, torch.device("cpu"), torch.float16, 0.001)
# log_probs = [[-10000., -60000., 0.]]
# entropy   = [0.]
```

Dividing by $\tau = 0.001$ in float32 produces $[60000, 10000, 70000]$ without issue. Because the largest finite float16 value is 65504, dividing in float16 first would overflow to $[60000, 10000, \infty]$ and turn the entire log-softmax row into NaN.

## Constraints

- $1 \le B \le 16$, $2 \le V \le 4096$, and $0.001 \le \tau \le 20$. Logits are finite and have magnitude at most 300.
- `log_probs` and `entropy` must be float32 and match the float64 reference within relative tolerance `1e-4` and absolute tolerance `1e-5`. Computing them in 16 bits misses this tolerance by two to three orders of magnitude.
- `model` must be called exactly once inside the autocast region, and the region must explicitly use `dtype` rather than the device default.
- For the test networks, log-probabilities of tokens with probability above $10^{-4}$ stay within 0.25 (bfloat16) and 0.05 (float16) of a full float32 forward pass. One test verifies this.
- **What runs where.** Standard cases run on the CPU, where `torch.autocast` supports both float16 and bfloat16 for linear layers. Two hardware-marked cases run only on an NVIDIA GPU on CUDA with `dtype=torch.float16`, including the low-temperature overflow case. The GPU cases were written from the PyTorch 2.15 documentation and could not be run when the problem was authored.

## Why this matters

Half precision cuts memory bandwidth in half for matrix multiplications and unlocks faster GPU tensor-core kernels, so inference servers run transformer bodies in float16 or bfloat16. The post-processing step has very different numerical needs. A low temperature such as 0.01 scales a logit of 700 to 70000, overflowing float16. In bfloat16, 8 significant bits quantize a log-probability near $-20$ to steps of 0.06. Summing across a 50,000-token vocabulary in 16 bits drops small terms that fall below the precision of the running total.

In production, keeping these operations in 16 bits causes NaN samples at low temperature, unstable top-p cutoffs, and perplexity drift against float32 baselines. Autocast only promotes a subset of operations on CUDA and stops protecting anything once the `with` block exits, so casting logits to float32 must be explicit.
