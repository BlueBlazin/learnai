Which operations in this pipeline are large matrix products, where 16 bits are enough
and the speed is won, and which are a handful of elementwise operations and sums over
one row of logits? What dtype does `torch.log_softmax(z, dim=-1)` return when `z` is a
bfloat16 tensor and no autocast region is active?

---

Use `torch.autocast` as a context manager around the single call `model(x)`. It needs
the *type* of the device as a string, which `device.type` gives you for a CPU and for
a CUDA device alike, and the 16-bit type as `dtype=`. Leave the region before doing
anything else. The logits are then a 16-bit tensor like any other: convert them with
`.float()` first, and only then divide by the temperature, take the log-softmax, and
sum for the entropy.

---

```python
with torch.autocast(device_type=device.type, dtype=dtype):
    logits = model(x)
logits = logits.float() / temperature
log_probs = torch.log_softmax(logits, dim=-1)
```

The entropy is `-(log_probs.exp() * log_probs).sum(dim=-1)`. Where a probability
underflows to 0 its log-probability is still a finite float32, and the product is 0.
