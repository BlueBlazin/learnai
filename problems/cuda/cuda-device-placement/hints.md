Which tensors does `run` receive, and which ones does it create itself? For every tensor it creates, consider which device it lands on if you omit `device=`, and what happens when it interacts with tensors on another device. For the padding mask, think about how to test whether position `t` is before the end of row `b` using a single tensor comparison without ever reading `lengths` into Python.

---

Write `to_device` as a recursive function with four cases: move a tensor with `.to(device)`, rebuild a `dict` from its items, rebuild a `list` or `tuple` with `type(batch)(...)`, and return any other value unchanged. Call `to_device` on the entire `inputs` dictionary at the start of `run`.

Next, create position IDs with `torch.arange(T, device=device)`, where `T = tokens.shape[1]` is a Python `int` that requires no device read. Broadcast the positions as a row against `lengths` as a column to form a `(B, T)` boolean mask on `device`. Avoid calling `.float()` on the mask or lengths, because multiplying a float tensor by a `bool` tensor and dividing by an integer tensor both preserve the float's dtype.

---

```python
positions = torch.arange(tokens.shape[1], device=device)
h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
valid = positions[None, :] < lengths[:, None]  # (B, T), bool, on the device
pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
```

Finally, compute the logits as `pooled @ weights["w_out"].T + inputs["head"]["bias"]`.
