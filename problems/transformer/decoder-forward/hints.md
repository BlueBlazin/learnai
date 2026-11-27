Before writing any code, list the tensors in `params` and check each one off as you
decide where it is used. For a two-layer tied model there are 21 tensors. Which one is
used twice? Which one is used once outside the loop and is the easiest to forget?

---

Structure the forward pass around a residual stream `x` of shape `(B, T, d_model)` that
is only ever updated by addition inside the layer loop:

1. Start with `x = embed[tokens]`.
2. For each layer, run the pre-norm block from the previous problem with multi-head
   attention and RoPE. After the projections, reshape `q` to
   `(B, T, n_heads, head_dim)` and move the head axis to position 1. Do the same for `k`
   and `v` using `n_kv_heads`. Only after splitting heads should you apply `ops.rope` to
   `q` and `k`, since its last axis must be one head (`head_dim`) and its second-to-last
   axis must be time. Run `ops.attention` with a lower-triangular boolean mask, then
   merge the heads back.
3. After the loop, apply `x = rms_norm(x, model.norm.weight)` once.
4. Multiply by the transposed head weight. Choose the matrix from `cfg.tie_embeddings`
   rather than checking what happens to be in `params`.

Pass `cfg.norm_eps` to every norm and `cfg.rope_base` to every rotation.

---

Here is the attention sublayer for one layer in PyTorch, where `p` is
`f"model.layers.{i}."`:

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

Use `positions = arange(T)` and `causal = tril(ones(T, T))` as booleans. The head weight
is `params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]`.
