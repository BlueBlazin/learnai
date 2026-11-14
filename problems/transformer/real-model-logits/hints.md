Count the tensors and the numbers. A tied 28-layer checkpoint has
$1 + 28 \cdot 8 + 1 = 226$ tensors, and `cfg` has seven numbers that are not shapes
(`norm_eps`, `rope_theta`, `attention_scale`, `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`, `tied_embeddings`). Your function is finished
when every tensor and every number is used. Which three of the numbers act *outside*
the blocks, and which tensor is used twice?

---

The frame, in order:

1. `x = embedding[tokens] * cfg.embedding_multiplier`.
2. `for i in range(cfg.n_layers)`: the block from `real-model-block`, with the head
   counts, `head_dim`, `d_ff`, `norm_eps`, `rope_theta`, `attention_scale`, and
   `residual_multiplier` read from `cfg` instead of written as literals. Loop over
   the integers; do not iterate over the names in `weights`. Build the RoPE tables and
   the causal mask once, before the loop.
3. `x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)`.
4. `head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]`.
5. `return (x @ head.T) / cfg.logits_scaling`.

If the small models pass and the real one does not, the difference is something only
28 layers can show. Print the names you index, in the order you index them.

---

When the logits are wrong, find the first place they go wrong instead of staring at
the end. The fixture generator shows how: `transformers` returns every intermediate
state with `output_hidden_states=True`, and the kit's `RealLM.hidden_states` gives the
same list for comparison outside the judge. The usual culprits, by symptom:

- logits about 4 times too large, argmax correct: `logits_scaling`;
- everything plausible but off by a few units, worse at later positions: the
  attention scale or the RoPE layout;
- logits enormous: the final norm is missing;
- right on a 3-layer model, wrong on 28 layers: the order of the layers
  (`"model.layers.10"` sorts before `"model.layers.2"`).
