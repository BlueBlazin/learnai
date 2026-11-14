Follow one number from `x` to the output. In the two equations of the specification,
how many times is it normalized on the way if both sublayers happen to output zero?
What does that tell you about which variable the norms may be assigned to?

---

Keep two kinds of variable apart: the **stream** (`x`, then `h`, then `y`), which only
ever changes by `stream = stream + something`, and the **sublayer input** (`u`), a
normalized copy that is thrown away once the sublayer has run.

1. `u = rms_norm(x, gain 1)`; project `u` to `q`, `k`, `v`; attend with the mask;
   project with `o_proj`; add the result to `x`. Call the sum `h`.
2. `u = rms_norm(h, gain 2)`, where `h` is the stream you just updated and gain 2 is
   the *other* gain; run the MLP on `u`; add the result to `h`.

The weights are `(out, in)`, so projecting a batch of row vectors is `u @ W.T`.

---

The attention sublayer in full:

```python
u = ops.rms_norm(x, params["input_layernorm.weight"])
q = u @ params["self_attn.q_proj.weight"].T
k = u @ params["self_attn.k_proj.weight"].T
v = u @ params["self_attn.v_proj.weight"].T
h = x + ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T
```

The MLP sublayer has the same three steps: normalize `h` with
`post_attention_layernorm.weight`, call `ops.swiglu` with the gate, up, and down
weights in that order, and return `h` plus the result.
