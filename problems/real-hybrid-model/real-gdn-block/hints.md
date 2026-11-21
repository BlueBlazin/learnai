Look at the numbers before writing code. Print the mean of
`input_layernorm.weight`, of `post_attention_layernorm.weight`, and of
`linear_attn.norm.weight` for the layer you were given. A gain that scales a
normalized vector should sit near one. Which of the three do, and what would you add
to the others so that they did?

---

The frame is four lines: norm, mixer, add; norm, MLP, add. The first and third norm are
`ops.rms_norm(u, 1 + w, 1e-6)`. Pass the epsilon every time: the default of
`ops.rms_norm` is `1e-5`, which is invisible on inputs of order one and half a percent
on hidden states of size 0.03.

The mixer is your `gated-delta-layer` with three changes: the nine tensors are looked
up under `p + "linear_attn."` (keep the trailing dot in `p`, or layer 1 also matches
layers 10 to 19), the sizes come from shapes (`H = A_log.shape[0]`,
`dv = norm.weight.shape[0]`, `dk = (rows of in_proj_qkv - H * dv) / (2 * H)`), and key
heads equal value heads, so nothing is repeated. Its own norm takes `norm.weight` as it
is stored.

---

When the small model passes and the real one does not, or the reverse, compare by
symptom:

- every entry off, already in row 0 of a new sequence: a norm weight (the `1 +`
  missing on a block norm, or added to the gated norm), the kernel direction, or the
  order of norm and gate;
- row 0 right and every later row off: the decay, `exp(-exp(A_log) * softplus(a +
  dt_bias))`. The first row of a sequence decays a memory that is still zero, so it
  cannot show a wrong decay;
- one call right and two calls wrong from the boundary on: the state. The first
  `K - 1` rows after the boundary need `conv_state` (rows of `P`, inputs of the
  convolution); every row after it needs `S`, shaped `(H, dv, dk)`;
- the small model right, real layer 13 right, real layer 1 wrong: the names you
  matched.
