## The idea

Porting this checkpoint requires tracking eleven architectural details, most of which cannot be inferred from tensor shapes alone. The table below lists each detail alongside its source.

| # | fact | where it comes from |
|---|---|---|
| 1 | embeddings are used as stored (no scale, no position embedding) | the code |
| 2 | 24 pre-norm blocks with attention at 3, 7, 11, 15, 19, 23 and Gated DeltaNet elsewhere | `layer_types` |
| 3 | block norms, final norm, `q_norm`, and `k_norm` use RMSNorm with $\epsilon = 10^{-6}$ and gain `1 + weight` | `rms_norm_eps`, the code |
| 4 | the norm inside a delta layer uses gain `weight` directly and runs before the gate | the code |
| 5 | delta layers have 16 key and 16 value heads of dimension 128 and conv kernel 4 | `linear_*` keys, shapes |
| 6 | attention has 8 query heads of dimension 256 on 2 KV heads, with `q_proj` interleaved per head as query then gate | `head_dim`, shapes, the code |
| 7 | RoPE uses base $10^7$ on the first 64 channels of each head, half-split within those 64 | `rope_parameters`, the code |
| 8 | attention scores are divided by $\sqrt{256}$ and outputs are multiplied by $\sigma(\text{gate})$ before `o_proj` | the code |
| 9 | SwiGLU MLP has intermediate width 3584 | `intermediate_size`, shapes |
| 10 | the output head is tied to the embedding matrix | `tie_word_embeddings`, no `lm_head` in file |
| 11 | weights are stored in bfloat16 on disk (except two small float32 tensor types) and computed in float32 | the safetensors header |

Here "the code" refers to `models/qwen3_5/modeling_qwen3_5.py` in Hugging Face Transformers 5.23, specifically `Qwen3_5RMSNorm` (zero-initialized weight applied as `1.0 + weight`), `Qwen3_5RMSNormGated` (one-initialized weight applied before the gate), `Qwen3_5Attention`, and `Qwen3_5GatedDeltaNet`.

Two structural choices stand out beyond the individual constants.

**The gate lives inside the query projection.** Gated attention requires one extra vector per head and token. Rather than storing a separate projection matrix, `q_proj` has twice as many rows, and the reference reshapes its output to `(T, H, 2d)` before splitting along the last axis. The shape `(4096, 1024)` by itself could mean 16 heads of 256, 8 heads of 512, or all queries followed by all gates. Only the per-head query-then-gate layout matches the fixture logits.

**Most of the model is recurrent, not attention.** Eighteen of the 24 mixers are Gated DeltaNet recurrences. For a single-call forward pass, the outer block structure looks the same. During autoregressive serving (`hybrid-decoder-forward`), however, this means maintaining 18 fixed-size recurrent states and only 6 growing KV caches.

## Reference solution (PyTorch)

```python
ROOT = "model.language_model."


def gated_attention(h, weights, prefix, cfg, positions, causal):
    steps, heads, d = h.shape[0], cfg.n_heads, cfg.head_dim
    both = (h @ weights[prefix + "q_proj.weight"].T).reshape(steps, heads, 2 * d)
    q, gate = both[..., :d], both[..., d:]                    # per head: query, then gate
    k = (h @ weights[prefix + "k_proj.weight"].T).reshape(steps, cfg.n_kv_heads, d)
    v = (h @ weights[prefix + "v_proj.weight"].T).reshape(steps, cfg.n_kv_heads, d)
    q = ops.rms_norm(q, 1.0 + weights[prefix + "q_norm.weight"], cfg.norm_eps).transpose(0, 1)
    k = ops.rms_norm(k, 1.0 + weights[prefix + "k_norm.weight"], cfg.norm_eps).transpose(0, 1)
    v = v.transpose(0, 1)
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
    attended = ops.attention(q, k, v, causal).transpose(0, 1).reshape(steps, heads * d)
    gated = attended * torch.sigmoid(gate.reshape(steps, heads * d))
    return gated @ weights[prefix + "o_proj.weight"].T


def qwen_logits(weights, cfg, tokens):
    def norm(u, name):
        return ops.rms_norm(u, 1.0 + weights[name], cfg.norm_eps)   # stored as gain - 1

    steps = tokens.shape[0]
    positions = torch.arange(steps)
    causal = positions[None, :] <= positions[:, None]
    table = weights[ROOT + "embed_tokens.weight"]
    x = table[tokens]
    for i, kind in enumerate(cfg.layers):
        p = f"{ROOT}layers.{i}."
        h = norm(x, p + "input_layernorm.weight")
        if kind == "linear_attention":
            x = x + delta_mixer(h, weights, p + "linear_attn.", cfg)
        else:
            x = x + gated_attention(h, weights, p + "self_attn.", cfg, positions, causal)
        h = norm(x, p + "post_attention_layernorm.weight")
        x = x + ops.swiglu(h, weights[p + "mlp.gate_proj.weight"],
                           weights[p + "mlp.up_proj.weight"], weights[p + "mlp.down_proj.weight"])
    x = norm(x, ROOT + "norm.weight")
    return x @ table.T
```

Here `delta_mixer` is the mixer from `real-gdn-block` initialized from zeros, reading its head counts and dimensions from `cfg`. The MLX reference uses the same structure with `swapaxes`. The JAX reference computes the same values with two minor rewrites for eager-mode performance. First, `u @ W.T` materializes a transposed copy of `W` (roughly 3 GB per forward pass on this model), so it computes `(W @ u.T).T` instead. Second, indexing `x[t]` inside the token loop dispatches a slice on every step, so it iterates directly over the arrays. Those two changes reduce a 12-token JAX forward pass from 0.43 s to 0.18 s, though a direct translation of the PyTorch code is equally correct. On the two fixture prompts, all three framework references agree with `transformers` within $6.1 \cdot 10^{-5}$ on logits up to magnitude 31, and their argmax predictions match at all 24 positions.

## Common mistakes

- **Omitting the output gate.** Half of `q_proj` goes unused, which an early tensor-count check catches.
- **Swapping query and gate, or splitting `q_proj` in half across heads.** Splitting `(2dH, D)` into all queries followed by all gates is a natural guess, but it puts all queries into the first half of the heads.
- **Applying RoPE to the entire head.** `ops.partial_rope` rotates every channel unless `cfg.partial_rotary_factor` is passed. With the factor, only 64 of 256 channels rotate.
- **Hardcoding RoPE hyperparameters.** The values `1e7` and `0.25` belong only to the real checkpoint. The second small test model uses $10^4$ and 0.5.
- **Using `q_norm`, `k_norm`, block norms, or the final norm as plain gains.** All of these store gain minus one. The final norm's stored weights average 3.3, so missing the `+ 1.0` shifts the scale dramatically.
- **Adding `1.0` to the delta layer's norm weight.** Unlike the other norms, `linear_attn.norm.weight` is stored as a plain gain.
- **Leaving the default epsilon in block norms.** `ops.rms_norm` defaults to $10^{-5}$, whereas the model uses $10^{-6}$. On a small random model with unit-scale activations, that difference can hide inside tolerance. On the real model, where hidden-state mean squares are near $10^{-3}$, 49 norms each slightly off shift the logits by tenths. The second small test model sets $\epsilon = 10^{-2}$ so this error fails without downloading the checkpoint.
- **Leaving the default epsilon in `q_norm` and `k_norm`.** In the real model, queries and keys are much larger than $\sqrt{10^{-5}}$ before normalization, so both real-model tests still pass. Only tests with a larger `cfg.norm_eps` expose this bug.
- **Scaling embeddings by $\sqrt{D}$.** Models like Gemma and the original Transformer scale embeddings, but Qwen3.5 does not.
- **Omitting the causal mask in attention layers.** The 18 delta layers are causal by construction, so only 6 layers leak future tokens, and the final token position still produces the right logits.
- **Iterating layers in lexicographical name order.** Sorting string keys produces `0, 1, 10, 11, ..., 19, 2, 20, ...`, which passes every model with 10 or fewer layers and silently fails on the 24-layer model.
- **Calling `ops.gated_delta_layer`.** That helper was permitted in `hybrid-decoder-forward`, but this problem requires implementing the delta layer directly.

## A reference that is not one

When two libraries implement the same model, either can act as a reference until they disagree. Measuring `mlx-lm` 0.32.2 against `transformers` 5.23.0 in float32 gives the following maximum logit differences:

| `mlx-lm` 0.32.2 | eight prompts, every logit (the kit's measurement) | three prompts of 12 tokens, 64 columns |
|---|---|---|
| as it loads the checkpoint | 0.70, another argmax at 5 of 86 positions | 0.26, none of 36 |
| after `model.set_dtype(mx.float32)` | 0.17, 1 of 86 | 0.07, none of 36 |
| weights cast to float32 before its `sanitize` | $1.7 \cdot 10^{-4}$, none | not measured |

The discrepancy happens where fact 3 meets fact 11. `mlx-lm` adds `1.0` to the offset norm weights once at load time while those weights are still in bfloat16, rounding every gain to 8 mantissa bits before any cast to float32 can occur (shifting the final norm's gains by up to 0.39%). This is the same error as computing in the storage dtype, just performed once during loading.

This problem allows an error of $10^{-3} + 10^{-3}\,|\text{logit}|$, which is 0.011 at a logit of 10 and 0.032 at the largest logits here (31). Even the smallest error in the first two rows (0.07) is six times the allowance at logit 10 and twice the largest allowance, so the test fixtures come from `transformers`. These numbers do not measure generation quality, but they show that a numerical tolerance only makes sense against a reference that performs the same float32 computation.

## Complexity

Each token requires roughly 22 million multiply-adds per delta block and 19 million multiply-adds per attention block (plus $O(T)$ per token for attention scores), totaling about 0.5 billion multiply-adds across all 24 blocks. The tied output head costs $248{,}320 \times 1024 = 254$ million multiply-adds per position, roughly equal to twelve transformer blocks. During autoregressive generation, only the last token position needs the output head.

## In the wild

Hugging Face `transformers` executes multi-token prefill calls through a chunked form of the delta rule and single-token cached steps through the recurrent form. On this model, its own two code paths differ by up to $1.5 \cdot 10^{-4}$ in the logits (measured for this curriculum), which is the same magnitude as the difference between `transformers` and a from-scratch port.

The checkpoint also includes a 12-layer vision tower and a one-layer multi-token-prediction head (`mtp.*`), neither of which is read by a text-only decoder (`mtp-chain-draft` covers that head). The delta layer comes from "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv:2412.06464). The gated attention, QK-norm, and offset norm storage were verified directly from the reference code.
