Keys, values, and scores must stay aligned along the sequence axis. Suppose you already had an index tensor `kept` of shape `(B, H, N)` containing the positions each head keeps. How would you produce all three outputs from `kept`?

---

Follow three steps:
1. Pad `scores` with `T` zeros on the right (or start from zeros if `scores` is `None`) and add the column sums `attn_weights.sum(axis=2)`.
2. If `S <= budget`, set `kept` to `0 .. S - 1` for every batch item and head. Otherwise split at `old = S - recent`, rank positions `0 .. old - 1` by score, take the top `budget - recent` indices, **sort those indices into increasing order**, and concatenate `old .. S - 1` after them.
3. Gather `k`, `v`, and `scores` along axis `2` using `kept`.

---

Use a stable descending sort so equal scores preserve their original position order (putting older positions first). In PyTorch, use `torch.argsort(x, descending=True, stable=True)`. In JAX and MLX, `argsort(-x)` is stable. Then:

```
ranked = argsort_descending(scores[:, :, :old])          # (B, H, old)
heavy  = sort(ranked[:, :, : budget - recent])           # back in sequence order
kept   = concatenate(heavy, arange(old, S) broadcast to (B, H, recent))
k_out  = take_along_axis(k, kept[..., None], axis=2)     # torch.gather needs the
scores = take_along_axis(scores, kept, axis=2)           # index expanded to (B, H, N, D)
```
