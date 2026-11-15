## The idea

`torch.autocast(device_type, dtype)` does not convert a model. It changes how
individual operations run while the region is active: an operation on autocast's
16-bit list (matrix products, `linear`, convolutions) casts its float32 inputs to
`dtype` on the way in, so the weights stay float32 in memory and the arithmetic is
16-bit. An operation on neither list runs in whatever dtype its inputs have, which
after the first linear layer is `dtype` too. The region ends with the `with` block.
After it, PyTorch is back to its ordinary rule (the output dtype follows the input
dtypes), and the logits are a 16-bit tensor that every later operation will happily
keep in 16 bits.

Three properties of the 16-bit types decide what may stay there:

| | float16 | bfloat16 | float32 |
|---|---|---|---|
| significant bits | 11 | 8 | 24 |
| largest finite value | 65504 | $3.4 \cdot 10^{38}$ | $3.4 \cdot 10^{38}$ |
| spacing of values near 20 | 0.016 | 0.125 | $1.9 \cdot 10^{-6}$ |

- **Range.** `logits / temperature` leaves the float16 range as soon as a logit
  exceeds $65504\,\tau$. The quotient is `inf`, the log-softmax subtracts `inf` from
  `inf`, and the row is NaN.
- **Resolution.** bfloat16 has the range of float32 and 8 significant bits. A scaled
  logit, and then a log-probability, is rounded to a grid of 0.125 near 20.
- **Accumulation.** Adding a term to a running sum that is $2^{8}$ (or $2^{11}$)
  times larger changes nothing. The normalizer and the entropy are sums over the
  vocabulary.

The matrix products tolerate all three, because their outputs go through a
normalization layer and their errors average out; the measured effect of a 16-bit
body on the test networks is an error of about 0.05 in the log-probabilities for
bfloat16 and 0.005 for float16. The statistics do not, and they are a negligible
share of the work. So: 16 bits inside the region, one explicit `.float()` at its
exit, float32 after it.

## Reference solution (PyTorch)

```python
@torch.inference_mode()
def forward_amp(model, x, device, dtype, temperature=1.0):
    with torch.autocast(device_type=device.type, dtype=dtype):
        logits = model(x)

    logits = logits.float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
```

`device.type` is `"cuda"` for `cuda:0` and `"cpu"` for the CPU, which is what
`device_type` wants. The entropy needs no special case for a probability of zero: the
log-probability is a finite float32 (the logits are finite), and $0 \cdot \ell = 0$.

Doing the float32 work *after* the region, and not merely after a `.float()` inside
it, matters for other code than this: inside a region an operation on the 16-bit list
casts float32 inputs back down, so `logits.float() @ w` under CPU autocast is a
bfloat16 product again. The documented way to force float32 for part of a region is
a nested `torch.autocast(device_type, enabled=False)` with the inputs cast by hand.

## What autocast does for you, and where

On CUDA, autocast has a second list: operations it runs in float32 even when their
inputs are float16. In PyTorch 2.15 that list contains `softmax`, `log_softmax`,
`layer_norm`, `sum`, `exp`, `log`, `pow`, `norm`, and the losses. On the CPU the
float32 list is a different one (pooling, FFTs, linear algebra, losses) and contains
neither `softmax` nor `log_softmax`; the documentation adds that float16 shares the
bfloat16 lists there. Two consequences:

- Inside a CUDA region, `torch.log_softmax(logits, -1)` returns float32 without your
  help. The division by the temperature just before it is not on the list (only the
  reflected `__rtruediv__` is), so `logits / temperature` is float16 and overflows
  first.
- The protection ends with the region. A `log_softmax` of a float16 tensor after the
  `with` block is a float16 operation on every device.

The CPU cases were run on torch 2.15, where the facts above about the CPU were
checked directly: a linear layer under `autocast("cpu", dtype=...)` returns the
16-bit type for both `bfloat16` and `float16`, and `log_softmax`, `/`, and `sum` keep
it. The statements about CUDA are taken from the op lists in the 2.15 documentation
("Automatic Mixed Precision package"); the GPU cases have not been run.

## Common mistakes

- **The whole pipeline inside the region**, with a `.float()` on the results. The
  numbers have already been computed in 16 bits; the cast only changes their label.
  At a low temperature in float16 they are NaN.
- **Returning 16-bit statistics.** Nothing raises: a bfloat16 `log_softmax` is a
  legal operation. The sampler downstream then works with 8 significant bits.
- **Dividing by the temperature before the cast**: `(logits / temperature).float()`.
  The quotient is rounded to 16 bits (an error of up to 0.4 percent in bfloat16) or
  overflows (float16).
- **The entropy on a 16-bit copy.** The log-softmax is right and the sum loses its
  small terms.
- **`device_type="cuda"` written out.** With the model on the CPU, autocast prints a
  warning ("CUDA is not available ... Disabling autocast"), and the body runs in
  float32. Nothing fails; the speedup is silently gone, and so is the agreement with
  the GPU deployment that the CPU test run was meant to check.
- **No `dtype=`.** The default is float16 on CUDA and bfloat16 on the CPU, so the
  same code computes different numbers on the two devices.
- **No region at all**, or **`model.to(dtype)` instead of a region.** The second one
  converts the caller's model in place: its parameters are 16-bit from then on for
  everyone who holds it, and every layer runs in 16 bits, including the
  normalizations that autocast would have kept in float32 on a GPU.

## Complexity

One forward pass, plus $O(BV)$ float32 work for the statistics: one widening copy of
the logits and a few elementwise passes over them.

## In the wild

Hugging Face Transformers loads models with `torch_dtype=torch.float16` or
`bfloat16`, and its `generate` loop converts the last position's logits to float32
before the logits processors and the sampler see them. T5 in float16 is the best-known case of a body
that itself overflows (its residual stream exceeds 65504 in some layers), which is
why bfloat16 is the usual choice on hardware that supports it: the same speed, the
range of float32, and a resolution that the body tolerates and the statistics do not.
