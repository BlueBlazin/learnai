There are three tensors that must stay aligned along the sequence axis: keys, values,
and scores. If you had the positions each head keeps, as an index tensor of shape
`(B, H, N)`, how would you produce all three outputs from it?

---

Work in this order. (1) Extend the scores with `T` zeros and add the column sums
`attn_weights.sum(axis=2)`. (2) If `S <= budget`, the kept positions are
`0 .. S - 1` for every head. Otherwise split the positions at `old = S - recent`:
rank positions `0 .. old - 1` by score, take the first `budget - recent` of that
ranking, **sort those positions**, and append `old .. S - 1`. (3) Gather keys,
values, and scores along axis 2 with the kept positions.

---

The ranking must put equal scores in position order, which is a stable descending
sort: `torch.argsort(x, descending=True, stable=True)`, or `argsort(-x)` in JAX and
MLX (both stable). Then

```
ranked = argsort_descending(scores[:, :, :old])          # (B, H, old)
heavy  = sort(ranked[:, :, : budget - recent])           # back in sequence order
kept   = concatenate(heavy, arange(old, S) broadcast to (B, H, recent))
k_out  = take_along_axis(k, kept[..., None], axis=2)     # torch.gather needs the
scores = take_along_axis(scores, kept, axis=2)           # index expanded to (B, H, N, D)
```
