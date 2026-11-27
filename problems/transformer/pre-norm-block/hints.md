Follow one number from `x` to the output. In the two equations of the specification,
how many times is it normalized on the way if both sublayers happen to output zero?
What does that tell you about which variable the norms may be assigned to?

---

Keep two kinds of variables separate. The **stream** (`x`, then `h`, then `y`) only
changes through additions of the form `stream = stream + something`. The **sublayer
input** (`u`) is a normalized copy that you discard once the sublayer finishes.

1. Compute `u = rms_norm(x, gain 1)` and project `u` to `q`, `k`, and `v`. Run
   attention with `mask`, project the output with `o_proj`, and add the result to `x`.
   Call that sum `h`.
2. Compute `u = rms_norm(h, gain 2)` using the updated stream `h` and the *other* gain.
   Run the MLP on `u` and add the result to `h`.

All linear weights have shape `(out, in)`, so projecting a batch of row vectors is
`u @ W.T`.

---

Here is the attention sublayer in full:

```python
u = ops.rms_norm(x, params["input_layernorm.weight"])
q = u @ params["self_attn.q_proj.weight"].T
k = u @ params["self_attn.k_proj.weight"].T
v = u @ params["self_attn.v_proj.weight"].T
h = x + ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T
```

The MLP sublayer follows the same three steps. Normalize `h` with
`post_attention_layernorm.weight`, call `ops.swiglu` with the gate, up, and down
weights in that order, and return `h` plus the result.
