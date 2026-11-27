Count the tensors before writing code. The text decoder uses 320 tensors in total: one embedding matrix, one final norm, 5 shared weights per block, plus 9 weights for each linear-attention layer and 6 for each full-attention layer ($2 + 18 \cdot 14 + 6 \cdot 11 = 320$). A complete forward pass reads each of those 320 tensors once per call. For the 49 block and final norm weights, the 2 extra norms in each attention layer, and the 1 extra norm in each delta layer, verify whether each weight is stored as a plain gain or as gain minus one.

---

Structure the forward pass in three stages:

1. Embed with `x = table[tokens]`.
2. Loop over layers with `for i, kind in enumerate(cfg.layers)` (iterate over integer indices rather than sorting keys from `weights`). In each block, run norm, mixer, and residual add, followed by norm, `ops.swiglu`, and residual add. Use your `real-gdn-block` mixer with `state=None` for `"linear_attention"`, keeping `norm.weight` as a plain gain.
3. Apply the final offset norm (`1 + weight`), then project with `x @ table.T`.

Write one helper function `norm(u, name)` for the offset RMSNorm and reuse it for all block norms, the final norm, `q_norm`, and `k_norm`. Always pass `cfg.norm_eps`.

For the `"full_attention"` mixer, track the tensor shape at every step before coding:

1. Project `q_proj` to `(T, 2 * d * H)`, reshape to `(T, H, 2 * d)`, and split the last axis at `d` into `q` and `gate`.
2. Project `k_proj` and `v_proj` and reshape each to `(T, G, d)`.
3. Apply the offset RMSNorm to `q` and `k` over their last axis.
4. Move the head axis to the front: `(H, T, d)` for `q`, and `(G, T, d)` for `k` and `v`.
5. Call `ops.partial_rope` on `q` and `k`, passing **both** `cfg.rope_base` and `cfg.partial_rotary_factor`.
6. Call `ops.attention` with a `(T, T)` lower-triangular boolean causal mask.
7. Transpose back to `(T, H, d)`, multiply elementwise by `sigmoid(gate)`, reshape to `(T, H * d)`, and apply `o_proj`.

---

If your logits do not match, locate the first diverging layer instead of debugging from the final output. Outside the judge, `RealHybrid(path).lm().hidden_states(ids)` returns the kit's hidden state after the embedding and after every block. Print the maximum absolute difference per layer against your own intermediate states:

- **Wrong from layer 0 onward:** Check the block norm (`1 + weight` and `cfg.norm_eps`) or the delta mixer. Layer 0 is a linear-attention layer, so you can isolate it with `real-gdn-block`.
- **Right through layer 2, wrong after layer 3:** The bug is in the attention mixer. The most common causes, in order, are splitting `q` and `gate` across heads instead of within each head (query first), missing `1 +` on `q_norm` and `k_norm`, rotating all channels instead of 64 of 256, and omitting `sigmoid(gate)`.
- **Right through layer 23, wrong logits:** Check the `1 +` on the final norm (its stored weights average 3.3, so the gain is around 4.3) or check that the output head uses the transposed embedding matrix.
- **Right on small models, wrong on the real model from a middle layer onward:** Check layer ordering (`"layers.10"` sorts before `"layers.2"` if you sort dictionary keys) and check that you are not using a default epsilon of `1e-5`.
