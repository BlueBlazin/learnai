## The idea

`torch.autocast(device_type, dtype)` does not modify a model's parameters. Instead, it controls per-operation precision while the context manager is active. Operations on autocast's lower-precision list (such as matrix multiplications, `linear`, and convolutions) cast float32 inputs to `dtype` before computing, while the weights stay float32 in memory. Operations not on any autocast list simply run in the dtype of their inputs, which becomes `dtype` after the first linear layer.

Once the `with` block exits, PyTorch returns to its normal promotion rules. Because the returned logits are already 16-bit, every downstream operation stays in 16 bits unless you cast explicitly.

Three numerical properties of 16-bit floats determine which operations are safe to keep in half precision:

| | float16 | bfloat16 | float32 |
|---|---|---|---|
| significant bits | 11 | 8 | 24 |
| largest finite value | 65504 | $3.4 \cdot 10^{38}$ | $3.4 \cdot 10^{38}$ |
| spacing of values near 20 | 0.016 | 0.125 | $1.9 \cdot 10^{-6}$ |

- **Range.** In float16, `logits / temperature` overflows to `inf` whenever a logit exceeds $65504\,\tau$. Subtracting `inf` from `inf` inside log-softmax then produces NaN for the entire row.
- **Resolution.** bfloat16 matches float32's exponent range but has only 8 significant bits. Near 20, adjacent representable values are 0.125 apart, coarsely quantizing scaled logits and log-probabilities.
- **Accumulation.** Adding a term smaller than $2^{-8}$ (bfloat16) or $2^{-11}$ (float16) of the running total changes nothing. Both the softmax normalizer and the entropy are sums over the entire vocabulary.

The network body tolerates these limits because normalization layers bound activations and rounding errors average out across many terms. On the test networks, running the body in 16 bits shifts log-probabilities by only about 0.05 (bfloat16) or 0.005 (float16). The final statistics cannot tolerate those errors, and they represent a tiny fraction of total compute. Running the forward pass inside `autocast`, exiting the block, and calling `.float()` once before scaling and reducing gets both the speed and the accuracy.

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

`device.type` returns `"cuda"` for `cuda:0` and `"cpu"` for CPU, which matches what `device_type` expects. The entropy calculation needs no special case for zero probabilities. Because the logits are finite, every float32 log-probability is finite, so any underflowed probability satisfies $0 \cdot \ell = 0$.

Exiting the autocast region before the float32 work is good practice beyond this specific function. Inside an active autocast region, operations on the 16-bit list cast float32 inputs back down (for example, `logits.float() @ w` under CPU autocast still runs in bfloat16). If you ever need float32 inside a larger autocast block, wrap that section in `torch.autocast(device_type, enabled=False)` and cast its inputs manually.

## What autocast does for you, and where

On CUDA, autocast also maintains a float32 promotion list for numerically sensitive operations. In PyTorch 2.15, that list includes `softmax`, `log_softmax`, `layer_norm`, `sum`, `exp`, `log`, `pow`, `norm`, and loss functions. On CPU, the float32 list is different (covering pooling, FFTs, linear algebra, and losses) and does not include `softmax` or `log_softmax`. The documentation notes that CPU float16 shares the bfloat16 op lists.

This creates two subtle behaviors:

- Inside a CUDA autocast region, `torch.log_softmax(logits, -1)` promotes to float32 automatically. However, dividing by temperature (`logits / temperature`) is not on the float32 list (only reflected `__rtruediv__` is), so the division runs in float16 and overflows before `log_softmax` ever sees the tensor.
- Automatic promotion ends as soon as the `with` block exits. Calling `log_softmax` on a float16 tensor outside autocast runs in float16 on every device.

The CPU cases were verified directly on PyTorch 2.15: a linear layer under `autocast("cpu", dtype=...)` returns the requested 16-bit type for both `bfloat16` and `float16`, and `log_softmax`, `/`, and `sum` preserve it. The CUDA details come from the PyTorch 2.15 "Automatic Mixed Precision package" documentation, and the GPU cases have not been run.

## Common mistakes

- **Running the entire pipeline inside autocast** and calling `.float()` only on the final outputs. The rounding error or overflow has already happened in 16 bits, and casting afterward only changes the tensor's dtype.
- **Returning 16-bit statistics.** PyTorch allows a bfloat16 `log_softmax` without error, leaving downstream sampling or metrics with 8 bits of precision.
- **Dividing by `temperature` before casting**, as in `(logits / temperature).float()`. The division rounds in 16 bits (up to 0.4% error in bfloat16) or overflows to `inf` in float16.
- **Computing entropy on a 16-bit tensor.** Even with accurate log-probabilities, summing in 16 bits drops small terms.
- **Hardcoding `device_type="cuda"`.** On CPU, PyTorch emits a warning ("CUDA is not available ... Disabling autocast") and silently runs the body in float32, masking precision differences during CPU testing.
- **Omitting `dtype=`.** Autocast defaults to `float16` on CUDA and `bfloat16` on CPU, causing the same code to behave differently across devices.
- **Omitting autocast or calling `model.to(dtype)`.** Calling `model.to(dtype)` mutates the caller's weights in place to 16 bits and forces every layer, including normalizations that CUDA autocast keeps in float32, to run in 16 bits.

## Complexity

The function performs one forward pass plus $O(BV)$ float32 work for the statistics, consisting of one widening cast of the logits and a few elementwise and reduction passes.

## In the wild

Hugging Face Transformers loads models in `torch.float16` or `torch.bfloat16`, and its `generate` loop casts the last-position logits to float32 before passing them to logits processors and samplers. T5 in float16 is a well-known example where even the model body overflows because its residual stream can exceed 65504. Hardware that supports bfloat16 avoids this overflow because bfloat16 pairs the throughput of float16 with the dynamic range of float32, though final probability statistics still require float32 casts.
