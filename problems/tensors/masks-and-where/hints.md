For each function, write down what a dropped entry should be replaced with so that
the rest of the computation can ignore the mask. Then ask which operation performs
that replacement without ever computing with the dropped value.

---

- `fill_masked` is a single `where`: the condition, the value where it is true, the
  value where it is false.
- `masked_mean`: a sum does not change when you add 0, so the dropped steps become 0.
  The mask is `(B, T)` and `x` is `(B, T, D)`; aligned from the right they do not
  match, so the mask needs a third axis. Divide by the number of kept steps of each
  row, which is a sum over the mask, and decide what a count of 0 should be divided
  by.
- `masked_max`: a max does not change when you add a candidate that loses to
  everything. A row that keeps nothing then needs no special case.

---

The key lines, in PyTorch (`jnp.where` and `mx.where` take the same arguments; use
`axis=` and `jnp.maximum(count, 1)` or `mx.maximum(count, 1)` there):

```
torch.where(keep, scores, float("-inf"))                         # fill_masked

kept = torch.where(mask[:, :, None], x, 0.0)                     # (B, T, D)
count = mask.sum(dim=1)[:, None]                                 # (B, 1)
kept.sum(dim=1) / count.clamp(min=1)                             # masked_mean

torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)      # masked_max
```
