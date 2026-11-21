Count before you code. The text decoder is 320 tensors: one embedding, one final norm,
and per layer five that every block has plus nine for a linear-attention layer or six
for an attention layer ($2 + 18 \cdot 14 + 6 \cdot 11 = 320$). Your function is
finished when it reads each exactly once per call. For each of the 49 norm weights
among them (the final norm, two per block), and for the two extra ones of every
attention layer and the one extra of every delta layer, decide: gain, or gain minus
one?

---

The frame:

1. `x = table[tokens]`.
2. `for i, kind in enumerate(cfg.layers)`: norm, mixer, add; norm, `ops.swiglu`, add.
   Loop over the integers; do not derive the order from the names in `weights`. The
   linear-attention mixer is your `real-gdn-block` mixer with `state=None`, with
   `norm.weight` as stored.
3. Final norm with `1 + weight`, then `x @ table.T`.

Write the offset norm once, as a function of `(u, name)`, and use it for the block
norms, the final norm, and the query and key norms. Pass `cfg.norm_eps` every time.

For the attention mixer, write the shape after every step before the code: `(T, 2dH)`
to `(T, H, 2d)`, cut the last axis at `d`; `(T, G, d)` for keys and values; norms on
the last axis; heads to the front, `(H, T, d)` and `(G, T, d)`; `ops.partial_rope`
with **both** `cfg.rope_base` and `cfg.partial_rotary_factor`; `ops.attention` with a
`(T, T)` lower-triangular boolean mask; back to `(T, H, d)`; times `sigmoid(gate)`;
merge; `o_proj`.

---

When the logits are wrong, find the first layer that is wrong instead of staring at
the end. Outside the judge, `RealHybrid(path).lm().hidden_states(ids)` gives the
kit's hidden state after the embedding and after every block; print the largest
difference per layer for your own intermediate states.

- Wrong from layer 0 on: a block norm (the `1 +`, the epsilon) or the delta mixer.
  Layer 0 is a linear-attention layer; check it alone with `real-gdn-block`.
- Right through layer 2, wrong after layer 3: the attention mixer. In order of
  likelihood: the query and gate cut (inside each head, query first), the `1 +` on
  `q_norm` and `k_norm`, the rotated share (64 of 256, not all), the missing
  `sigmoid(gate)`.
- Right through layer 23 and wrong logits: the final norm's `1 +` (its stored weights
  average 3.3, so the gain is about 4.3), or a head that is not the embedding matrix.
- Right on the small models, wrong on the real one from some middle layer on: the
  order of the layers (`"layers.10"` sorts before `"layers.2"`), or an epsilon of
  `1e-5`.
