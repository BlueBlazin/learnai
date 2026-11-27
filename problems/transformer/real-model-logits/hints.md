Count the tensors and config values. A tied 28-layer checkpoint has
$1 + 28 \cdot 8 + 1 = 226$ tensors, and `cfg` has seven non-shape fields
(`norm_eps`, `rope_theta`, `attention_scale`, `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`, `tied_embeddings`). Your implementation is
complete when every tensor and every field is used. Which three fields act *outside*
the transformer blocks, and which tensor is used twice?

---

Assemble the outer frame in order:

1. `x = embedding[tokens] * cfg.embedding_multiplier`.
2. `for i in range(cfg.n_layers)`: run the block from `real-model-block`, reading the
   head counts, `head_dim`, `d_ff`, `norm_eps`, `rope_theta`, `attention_scale`, and
   `residual_multiplier` from `cfg` instead of hardcoding them. Loop over the integer
   layer indices rather than iterating over keys in `weights`. Precompute the RoPE
   tables and causal mask once before the loop.
3. `x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)`.
4. `head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]`.
5. `return (x @ head.T) / cfg.logits_scaling`.

If the small models pass but the real model fails, look for a bug that only appears
with more than 10 layers. Print the tensor names you access in the exact order you
access them.

---

When the logits do not match, find the first layer where the activations diverge
instead of inspecting only the final output. In `transformers`, passing
`output_hidden_states=True` returns every intermediate state, and the kit's
`RealLM.hidden_states` provides the same list for local debugging outside the judge.

Common symptoms and their causes:

- **Logits are about 4 times too large, but the argmax is right.** Check
  `logits_scaling`.
- **Logits look plausible but are off by a few units, worse at later positions.**
  Check `attention_scale` and the RoPE half-split layout.
- **Logits are enormous.** Check that you applied the final RMSNorm with
  `model.norm.weight`.
- **The 3-layer models pass, but the 28-layer model fails.** Check the layer order
  (`"model.layers.10"` sorts before `"model.layers.2"` in string order).
