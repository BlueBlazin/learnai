The state after a call depends only on how many tokens have been appended: the newest
`keep_recent` are float, everything older is codes. So what is "everything that is
float after the new chunk arrives", and how many of its oldest tokens have to leave?

---

Concatenate the old float window and the new chunk along the sequence axis. If that
has `n` tokens, the oldest `leaving = max(0, n - keep_recent)` of them are quantized
now and the rest is the new window. `leaving` may reach into the new chunk, and it
may be 0: slicing `[:, :, :0]` gives a tensor with an axis of length 0, which
quantizes to codes, scales, and biases of length 0 and concatenates like any other.
The codes already in the cache are only concatenated, never recomputed.

---

Quantizing a slab `x` of shape `(B, H, S, D)`:

```
groups = x.reshape(B, H, S, D // group_size, group_size)
lo, hi = groups.min(-1), groups.max(-1)                 # (B, H, S, D // group_size)
scale  = where(hi > lo, (hi - lo) / (2**bits - 1), 1)
codes  = clamp(round((groups - lo[..., None]) / scale[..., None]), 0, 2**bits - 1)
```

then cast the codes to uint8 and reshape them back to `(B, H, S, D)`; `bias` is `lo`.
Reading is the mirror image: reshape the codes to
`(B, H, S, G, D // G)` with `G = scale.shape[-1]`, cast to float32, multiply by
`scale[..., None]`, add `bias[..., None]`, reshape back, and concatenate the float
window behind it. Spell out `D // G` in that reshape: a `-1` cannot be inferred when
`S` is 0.
