For each function, decide what value should replace a masked-out entry so the rest of the computation can ignore the mask. Then choose an operation that performs that replacement without doing arithmetic on the dropped value.

---

- `fill_masked` is a single `where` call taking the condition, the value when true, and the value when false.
- `masked_mean`: adding `0` leaves a sum unchanged, so replace dropped steps with `0`. Because `mask` has shape `(B, T)` and `x` has shape `(B, T, D)`, right-aligned broadcasting will not match them until you add a third axis to `mask`. Divide by the per-row count of kept steps (`mask.sum(...)`) and guard against dividing by `0`.
- `masked_max`: a maximum is unchanged by any value that loses to all finite numbers. Using that fill value also handles rows with no kept steps without a special case.

---

The key lines in PyTorch (`jnp.where` and `mx.where` take the same arguments; use `axis=` and `jnp.maximum(count, 1)` or `mx.maximum(count, 1)` there):

```
torch.where(keep, scores, float("-inf"))                         # fill_masked

kept = torch.where(mask[:, :, None], x, 0.0)                     # (B, T, D)
count = mask.sum(dim=1)[:, None]                                 # (B, 1)
kept.sum(dim=1) / count.clamp(min=1)                             # masked_mean

torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)      # masked_max
```
