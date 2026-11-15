Write the content score of one head as a chain of matrices acting on the two
latents: $q^C_h \cdot k^C_h = (W^{UQ}_h c^Q)^\top (W^{UK}_h c)$. Everything between
$c^Q$ and $c$ is constant, the same for every token. What single matrix is it, and
which side would you rather apply it to: the one query of this step or all $S$
cached latents? Then do the same for $W^O_h W^{UV}_h$ acting on a weighted sum of
latents.

---

`absorb`: reshape the up-projections to `(H, d_head, ...)` so that head `h` is
`[h]`, and `w_o` to `(d_model, H, d_head)` then move the head axis first. The two
results are batched matrix products: `uk^T @ uq` and `o @ uv`.

`mla_step`: compute `c_q = x @ w_dq.T`. The content query in latent space is
`w_qk[h] @ c_q` for each head, shape `(B, H, T, d_c)`; the RoPE query is
`c_q @ w_qr.T` reshaped to `(B, H, T, d_rope)` and rotated. New cache entries are
`x @ w_dkv.T` and the rotated `x @ w_kr.T`; concatenate them behind the old ones
along axis 1. The score is the sum of two matrix products against the two cached
arrays, each with a head axis of size one so that all heads share them. The
positions of the new tokens are `S, S + 1, ...`, and the mask is the one from
`kv-cache-append`.

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

The scale uses `d_head`, a number you can no longer read off any shape: the model
was trained with queries and keys of width `d_head + d_rope`, and absorbing a
matrix does not change the scores.
