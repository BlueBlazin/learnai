`load_weights` matches a tensor called `model.layers.3.self_attn.q_proj.weight` to
the parameter it finds by walking those names through the module: attribute `model`,
attribute `layers`, list index 3, attribute `self_attn`, attribute `q_proj`,
parameter `weight`. So the question for the constructor is not "which layers does
the model have" but "which attribute names and which nesting spell the checkpoint's
names"? And which names must be absent for a tied model?

---

Mirror the names with small modules: a top-level `Granite` holding `self.model` (and
`self.lm_head = nn.Linear(...)` only when the config is untied); a backbone with
`embed_tokens` (`nn.Embedding`), `layers` (a Python list of blocks), and `norm`
(`nn.RMSNorm`); a block with `input_layernorm`, `self_attn`, `post_attention_layernorm`,
and `shared_mlp`; an attention module with `q_proj`, `k_proj`, `v_proj`, `o_proj`
(`nn.Linear(..., bias=False)`); an MLP with `input_linear` and `output_linear`.
Keep the multipliers as Python floats on the modules: a float is an attribute, not a
parameter, and multiplying by it keeps the array's dtype. `nn.RoPE` has no
parameters, so it can be an attribute of the attention module without adding a name.

---

The attention forward pass, in the three fused kernels:

```python
q = self.rope(
    heads(self.q_proj(x), self.n_heads)
)  # nn.RoPE(head_dim, traditional=False, base=cfg.rope_theta)
k = self.rope(heads(self.k_proj(x), self.n_kv_heads))
v = heads(self.v_proj(x), self.n_kv_heads)
mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.attention_scale, mask="causal")
out = self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(B, T, -1))
```

with `heads(z, n) = z.reshape(B, T, n, -1).transpose(0, 2, 1, 3)`. Do not repeat the
key and value heads: the kernel handles grouped-query attention when `k` and `v` have
fewer heads than `q`. The MLP is `gate, up = mx.split(self.input_linear(h), 2,
axis=-1)` and `self.output_linear(nn.silu(gate) * up)`. The tied head is
`self.model.embed_tokens.as_linear(x)`.
