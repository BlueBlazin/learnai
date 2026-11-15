Which tensors does `run` receive, and which does it have to make? For every tensor it
makes, ask where it will live if you say nothing, and what happens when it meets a
tensor that lives somewhere else. For the mask: can you say "position `t` is before
the end of row `b`" as one comparison between two tensors, without ever asking what
the lengths are?

---

`to_device` is a recursion with four cases: a tensor (move it), a dict (rebuild it
from its items), a list or tuple (rebuild it with `type(batch)(...)`), anything else
(return it). Call it first in `run`, on the whole of `inputs`.

Then create the position ids with `torch.arange(T, device=device)`; `T` comes from
`tokens.shape[1]`, which is a Python int and costs nothing to read. Broadcasting the
positions as a row against the lengths as a column gives the `(B, T)` mask on the
device. Do not convert the mask or the lengths with `.float()`: multiplying a float
tensor by a boolean one, and dividing it by an integer one, keeps the float's dtype.

---

```python
positions = torch.arange(tokens.shape[1], device=device)
h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
valid = positions[None, :] < lengths[:, None]  # (B, T), bool, on the device
pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
```

The logits are `pooled @ weights["w_out"].T` plus the bias from the moved inputs.
