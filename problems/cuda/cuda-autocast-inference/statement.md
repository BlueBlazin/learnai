Run a model's forward pass in mixed precision and turn its logits into
log-probabilities and an entropy. The body of the network is fast and accurate enough
in a 16-bit float type; the last few operations are not, and nothing in PyTorch stops
them from running in 16 bits once the logits are 16-bit.

## Specification

```python
def forward_amp(model, x, device, dtype, temperature=1.0): ...  # returns (log_probs, entropy)
```

- `model`: a module whose parameters are float32 and on `device`. `model(x)` returns
  logits of shape `(B, V)`.
- `x`: a float32 tensor of shape `(B, D)`, already on `device`.
- `device`: the device of the model and of `x`, `torch.device("cuda")` or
  `torch.device("cpu")`.
- `dtype`: the 16-bit type of the mixed-precision region, `torch.float16` or
  `torch.bfloat16`.
- `temperature`: a positive Python float, $\tau$.

The function does two things, in two precisions.

1. **The body, in mixed precision.** Call `model(x)` exactly once, inside an autocast
   region for the device type of `device` with the 16-bit type `dtype`. Inside such a
   region PyTorch runs matrix products and linear layers in `dtype` although the
   parameters stay float32. The logits $z$ that come out are 16-bit.
2. **The statistics, in float32.** Convert the logits to float32 and compute, in
   float32 throughout:

$$
\ell_{b,i} = \frac{z_{b,i}}{\tau} - \log \sum_{j} e^{z_{b,j} / \tau},
\qquad
H_b = -\sum_{i} e^{\ell_{b,i}}\, \ell_{b,i}.
$$

   The division by $\tau$ is a float32 division of the float32 logits.

Return the tuple `(log_probs, entropy)`: $\ell$ of shape `(B, V)` and $H$ of shape
`(B,)`, both float32 and on `device`.

The model must not be modified: after the call its parameters are still float32.
Gradients are not needed and not checked.

**What the judge observes.** The model records, at each call, whether an autocast
region was active for its device, which 16-bit type the region used, and the logits it
returned. The expected statistics are computed in float64 from exactly those logits,
so the only error left to you is float32 rounding.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, D)` | float32 |
| `model(x)` inside the region | `(B, V)` | `dtype` (float16 or bfloat16) |
| `log_probs` | `(B, V)` | float32 |
| `entropy` | `(B,)` | float32 |

## Example

The model is one linear layer without bias, with weight rows $[40, 0]$, $[0, 40]$,
$[40, 40]$, and `x = [[1.5, 0.25]]`. Its logits are $[60, 10, 70]$, which float16 and
bfloat16 both represent exactly.

```python
forward_amp(model, x, torch.device("cpu"), torch.bfloat16, 20.0)
# log_probs = [[-1.0046, -3.5046, -0.5046]]
# entropy   = [0.7779]
```

The scaled logits are $[3, 0.5, 3.5]$, and
$\log(e^{3} + e^{0.5} + e^{3.5}) = \log 54.850 = 4.0046$. The probabilities are
$[0.3662, 0.0301, 0.6038]$, so
$H = 0.3662 \cdot 1.0046 + 0.0301 \cdot 3.5046 + 0.6038 \cdot 0.5046 = 0.7779$.

```python
forward_amp(model, x, torch.device("cpu"), torch.float16, 0.001)
# log_probs = [[-10000., -60000., 0.]]
# entropy   = [0.]
```

The scaled logits are $[60000, 10000, 70000]$. In float32 that is unremarkable. The
largest finite float16 is 65504, so a division carried out on the float16 logits gives
$[60000, 10000, \infty]$, and the log-softmax of that row is NaN in every position.

## Constraints

- $1 \le B \le 16$, $2 \le V \le 4096$, $0.001 \le \tau \le 20$. The logits are finite
  and at most 300 in magnitude.
- `log_probs` and `entropy` must be float32 and are compared to a relative tolerance
  of `1e-4` and an absolute tolerance of `1e-5`. Arithmetic done in a 16-bit type
  misses that by two to three orders of magnitude.
- The model is called exactly once, inside the region; the region's type must be
  `dtype`, not the device's default.
- For the test networks, the log-probabilities of tokens with probability above
  $10^{-4}$ are within 0.25 (bfloat16) and 0.05 (float16) of a full float32 forward
  pass; one case checks it.
- **What runs where.** The cases without a hardware mark run on the CPU, where
  autocast accepts both 16-bit types and makes linear layers compute in them. Two
  cases run only on a machine with an NVIDIA GPU: the same checks with the model on
  the CUDA device and `dtype=torch.float16`, including the temperature that overflows
  float16. The GPU cases were written from the PyTorch 2.15 documentation and could
  not be run when the problem was authored.

## Why this matters

Half precision halves the memory traffic of every matrix product and lets the GPU use
its fastest kernels, which is why inference servers run the transformer body in
float16 or bfloat16. The end of the pipeline is different: a temperature of 0.01 turns
a logit of 700 into 70000, past the float16 maximum; bfloat16 keeps 8 significant
bits, so a log-probability near $-20$ is only known to within 0.06; and a sum over a
50,000-token vocabulary in 16 bits drops every term that is small against the running
total. The symptoms are NaN samples at low temperature, top-p cutoffs that move from
run to run, and perplexities that do not match the float32 baseline. Autocast protects
some of these operations on CUDA and none of them once the code has left the region,
so the conversion to float32 has to be written down.
