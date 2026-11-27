Write a single helper function that rotates the trailing `rope_dim` channels of a
tensor of shape `(..., N, d)` using a vector of `N` positions. Count how many times the
specification calls it, and check which three tensors and position vectors it takes.
Remember that positions can be negative, and rotating by $-p$ undoes a rotation by $p$.

---

To build the helper, split `x` into `x[..., :d - rope_dim]` (unchanged) and
`x[..., d - rope_dim:]` (the rotary slice). Inside the slice, pair adjacent channels
`tail[..., 0::2]` and `tail[..., 1::2]`, and compute frequencies as
`base ** (-arange(0, rope_dim, 2) / rope_dim)` with `rope_dim` in the denominator, not
`d`. Apply the 2D rotation, interleave the rotated pairs back together (stack on a new
last axis and reshape), and concatenate the unchanged channels in front.

In `shared_kv_attention`, rotate `q` by `pos_q` and `kv` by `pos_k`. Use the rotated
`kv` twice: as the key in `q @ kv.T / sqrt(d)` and as the value in `weights @ kv`.
Finally, rotate the attention output back by `-pos_q`.

---

```
q  = rot(q,  pos_q)
kv = rot(kv, pos_k)
scores  = q @ kv.T / sqrt(d)            # (H, T, S)
scores  = where(mask, scores, -inf)
out     = softmax(scores, -1) @ kv      # (H, T, d)
return rot(out, -pos_q)
```

The final rotation works because rotation is linear. Turning the weighted sum by $-p_t$
turns every term by $-p_t$, so a cached row rotated by $p_s$ arrives rotated by
$p_s - p_t$.
