`load_weights` maps a checkpoint tensor like `model.layers.3.self_attn.q_proj.weight` by walking attributes and list indices through your module: `self.model`, `.layers[3]`, `.self_attn`, `.q_proj`, `.weight`. Design your submodule hierarchy and attribute names directly from the checkpoint tensor names, and check which parameter must be omitted when `cfg.tied_embeddings` is `True`.

---

Mirror the checkpoint structure with small submodules:

- `Granite` holds `self.model` and adds `self.lm_head = nn.Linear(...)` only when `cfg.tied_embeddings` is `False`.
- The backbone holds `embed_tokens` (`nn.Embedding`), `layers` (a Python list of blocks), and `norm` (`nn.RMSNorm`).
- Each block holds `input_layernorm`, `self_attn`, `post_attention_layernorm`, and `shared_mlp`.
- The attention submodule holds `q_proj`, `k_proj`, `v_proj`, and `o_proj` (`nn.Linear(..., bias=False)`), plus `nn.RoPE` (which has no parameters, so it adds no entry to the parameter tree).
- The MLP holds `input_linear` and `output_linear`.

Store multipliers as plain Python floats so they do not become parameters and do not upcast 16-bit arrays.

---

Inside attention, given input shape `(B, T, d_model)` and helper `heads(z, n) = z.reshape(B, T, n, -1).transpose(0, 2, 1, 3)`, compose the fused kernels like this:

```python
q = self.rope(
    heads(self.q_proj(x), self.n_heads)
)  # nn.RoPE(head_dim, traditional=False, base=cfg.rope_theta)
k = self.rope(heads(self.k_proj(x), self.n_kv_heads))
v = heads(self.v_proj(x), self.n_kv_heads)
mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.attention_scale, mask="causal")
out = self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(B, T, -1))
```

Do not repeat the key and value heads, because the kernel handles grouped-query attention directly. For the MLP, use `gate, up = mx.split(self.input_linear(x), 2, axis=-1)` followed by `self.output_linear(nn.silu(gate) * up)`. For a tied head, use `self.model.embed_tokens.as_linear(x)`.
