Before writing any code, list the tensors in `params` and tick each one off as you
decide where it is used. For a two-layer tied model there are 21 of them. Which one is
used twice? Which one is used once, outside the loop, and is the easiest to forget?

---

Build the pass as a stream `x` of shape `(B, T, d_model)` that is only ever added to:

1. `x = embed[tokens]`.
2. For each layer: the pre-norm block from the previous problem. The new part is
   inside attention. After the projections, reshape `q` to `(B, T, n_heads, head_dim)`
   and move the head axis to position 1; do the same for `k` and `v` with
   `n_kv_heads`. Only now apply `ops.rope` to `q` and `k`: its last axis must be one
   head (`head_dim` wide), and its second-to-last axis must be time. Then
   `ops.attention` with a lower-triangular boolean mask, then undo the reshape.
3. `x = rms_norm(x, model.norm.weight)`, once, after the loop.
4. Multiply by the transposed head. Which matrix that is depends on
   `cfg.tie_embeddings`, not on what happens to be in the dict.

Pass `cfg.norm_eps` to every norm and `cfg.rope_base` to every rotation.

---

The attention sublayer of one layer, in PyTorch (`p` is the prefix
`f"model.layers.{i}."`):

```python
def heads(t, n):  # (B, T, n * head_dim) -> (B, n, T, head_dim)
    return t.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)

h = ops.rms_norm(x, params[p + "input_layernorm.weight"], cfg.norm_eps)
q = heads(h @ params[p + "self_attn.q_proj.weight"].T, cfg.n_heads)
k = heads(h @ params[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
v = heads(h @ params[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
q, k = ops.rope(q, positions, cfg.rope_base), ops.rope(k, positions, cfg.rope_base)
attended = ops.attention(q, k, v, causal)            # (B, n_heads, T, head_dim)
attended = attended.transpose(1, 2).reshape(batch, seq, cfg.d_model)
x = x + attended @ params[p + "self_attn.o_proj.weight"].T
```

with `positions = arange(T)` and `causal = tril(ones(T, T))` as booleans. The head is
`params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]`.
