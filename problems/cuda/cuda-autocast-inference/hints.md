Which operations in this pipeline are large matrix products that benefit from 16-bit speed, and which are elementwise ops and reductions over one row of logits? What dtype does `torch.log_softmax(z, dim=-1)` return when `z` is a bfloat16 tensor and no autocast region is active?

---

Wrap the single call `model(x)` in a `torch.autocast` context manager. Pass `device_type=device.type` (which works for both CPU and CUDA devices) and `dtype=dtype`. Exit the `with` block before doing any post-processing. The returned `logits` are still 16-bit, so call `.float()` first, then divide by `temperature`, compute `log_softmax`, and sum for the entropy.

---

```python
with torch.autocast(device_type=device.type, dtype=dtype):
    logits = model(x)
logits = logits.float() / temperature
log_probs = torch.log_softmax(logits, dim=-1)
```

Compute the entropy as `-(log_probs.exp() * log_probs).sum(dim=-1)`. Even when a probability underflows to 0, its float32 log-probability remains finite, so the product is 0.
