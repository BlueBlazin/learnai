Write one helper that rotates the trailing `rope_dim` channels of a tensor of shape
`(..., N, d)` by a vector of `N` positions, and count how many times the definition
uses it. Which three tensors does it apply to, and with which positions? Note that a
position can be negative: rotating by $-p$ is the inverse of rotating by $p$.

---

The helper: slice off `x[..., :d - rope_dim]` (unchanged) and `x[..., d - rope_dim:]`
(the slice). Inside the slice, the pairs are adjacent channels, `tail[..., 0::2]` and
`tail[..., 1::2]`, and the frequencies are `base ** (-arange(0, rope_dim, 2) /
rope_dim)`: the denominator is `rope_dim`, not `d`. Rotate, interleave the two halves
again (stack on a new last axis, then reshape), and concatenate the unchanged channels
in front.

Then the function is: rotate `q` by `pos_q`, rotate `kv` by `pos_k`, and use the
rotated `kv` twice, as the key in `q @ kv.T / sqrt(d)` and as the value in
`weights @ kv`.

---

```
q  = rot(q,  pos_q)
kv = rot(kv, pos_k)
scores  = q @ kv.T / sqrt(d)            # (H, T, S)
scores  = where(mask, scores, -inf)
out     = softmax(scores, -1) @ kv      # (H, T, d)
return rot(out, -pos_q)
```

The last line works because rotation is linear: turning the weighted sum by
$-p_t$ turns every term by $-p_t$, and a row that was turned by $p_s$ ends up turned
by $p_s - p_t$.
