Write one head's content attention score in terms of the two latents: $q^C_h \cdot k^C_h = (W^{UQ}_h c^Q)^\top (W^{UK}_h c)$. Everything between $c^Q$ and $c$ is a constant matrix that does not depend on the token. What single matrix is it, and is it cheaper to apply it to the new query or to all $S$ cached latents? Apply the same factoring to $W^O_h W^{UV}_h$ acting on a weighted sum of latents.

---

For `absorb`, reshape `w_uq`, `w_uk`, and `w_uv` to `(H, d_head, ...)` so axis 0 indexes heads. Reshape `w_o` to `(d_model, H, d_head)` and move the head axis to the front. The absorbed weights are then the batched products `uk^T @ uq` and `o @ uv`.

For `mla_step`, compute `c_q = x_new @ w_dq.T`. Project `c_q` into latent space per head with `w_qk` to get shape `(B, H, T, d_c)`, and project `c_q @ w_qr.T`, reshape to `(B, H, T, d_rope)`, and apply `ops.rope`. For the cache, compute `x_new @ w_dkv.T` and the rotated `x_new @ w_kr.T`, then append them to the previous cache along axis 1. Score all heads against the shared cached tensors by inserting a head axis of size 1. New tokens occupy positions `S, S + 1, ...`, and the causal mask matches `kv-cache-append`.

---

```
positions = S + arange(T)
c      = cat(cache_c,  x @ w_dkv.T)                      # (B, S+T, d_c)
k_rope = cat(cache_kr, rope(x @ w_kr.T, positions))      # (B, S+T, d_r)
c_q    = x @ w_dq.T
q_lat  = einsum("hcq,btq->bhtc", w_qk, c_q)              # (B, H, T, d_c)
q_rope = rope((c_q @ w_qr.T).reshape(B, T, H, d_r).swap(1, 2), positions)
scores = (q_lat @ c.T[:, None] + q_rope @ k_rope.T[:, None]) / sqrt(d_head + d_r)
probs  = softmax(where(j <= S + i, scores, -inf))        # (B, H, T, S+T)
mixed  = probs @ c[:, None]                              # (B, H, T, d_c): latents, not values
out    = einsum("hdc,bhtc->btd", w_ov, mixed)
```

Use `d_head` from `weights` in the scale factor even though it no longer appears in any tensor shape. The model was trained with queries and keys of width `d_head + d_rope`, and absorbing the projections does not change the scores.
