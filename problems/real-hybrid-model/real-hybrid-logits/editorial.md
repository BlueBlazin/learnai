## The idea

The model is a list of facts, and the checkpoint states almost none of them. This is
the list; the right column says where each one can be read.

| # | fact | where it comes from |
|---|---|---|
| 1 | embeddings are used as stored: no scale, no position embedding | the code |
| 2 | 24 pre-norm blocks; attention at 3, 7, 11, 15, 19, 23, Gated DeltaNet elsewhere | `layer_types` |
| 3 | block norms, final norm, `q_norm`, `k_norm`: RMSNorm with $\epsilon = 10^{-6}$ and gain `1 + weight` | `rms_norm_eps`; the code |
| 4 | the norm inside a delta layer: gain `weight`, norm before gate | the code |
| 5 | delta layers: 16 key and 16 value heads of 128, kernel 4 | `linear_*` keys, shapes |
| 6 | attention: 8 heads of 256 on 2 key/value heads; `q_proj` is per head query then gate | `head_dim`, shapes; the code |
| 7 | RoPE with base $10^7$ on the first 64 channels of a head, half-split inside them | `rope_parameters`; the code |
| 8 | scores divided by $\sqrt{256}$; output times $\sigma(\text{gate})$ before `o_proj` | the code |
| 9 | SwiGLU of width 3584 | `intermediate_size`, shapes |
| 10 | the head is the embedding matrix | `tie_word_embeddings`; no `lm_head` in the file |
| 11 | weights are bfloat16 on disk (two kinds of small tensor float32), used as float32 | the safetensors header |

"The code" is `models/qwen3_5/modeling_qwen3_5.py` of Hugging Face Transformers 5.23:
`Qwen3_5RMSNorm` (a zero-initialized weight applied as `1.0 + weight`),
`Qwen3_5RMSNormGated` (a one-initialized weight, "norm before gate"),
`Qwen3_5Attention`, and `Qwen3_5GatedDeltaNet`.

Two structural points are worth more than the constants.

**The gate lives in the query projection.** Gated attention needs one extra vector per
head and token. Instead of a separate matrix, `q_proj` is twice as tall, and the
reference reshapes its output to `(T, H, 2d)` and chunks the last axis. The shape
`(4096, 1024)` alone is consistent with 16 heads of 256, with 8 heads of 512, with
queries followed by gates, and with what the model does. Only the last gives the
fixture's logits.

**Most of the model is not attention.** 18 of 24 mixers are recurrences. For logits
from one call that changes nothing about the frame; for serving it changes the cache
(`hybrid-decoder-forward`): 18 fixed-size states and 6 growing key-value caches.

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

`delta_mixer` is the mixer of `real-gdn-block` started from zeros, with the sizes read
from `cfg`. The MLX version is the same lines with `swapaxes`. The JAX reference
computes the same thing in two other spellings, because eager JAX charges for things
the other two give away: `u @ W.T` materializes a transposed copy of `W` (3 GB per
forward pass of this model), so it multiplies from the weight's side, `(W @ u.T).T`;
and every `x[t]` in the token loop is a dispatched slice, so it iterates over the
arrays instead. That takes its forward pass of 12 tokens from 0.43 s to 0.18 s; a
solution written like the PyTorch one is just as correct. On the two
prompts of the fixture the three references agree with `transformers` to within
$6.1 \cdot 10^{-5}$ on logits of magnitude up to 31, and the argmax is equal at all 24
positions.

## Common mistakes

- **No output gate.** Half of `q_proj` is then unused, which a tensor count would have
  caught.
- **Query and gate exchanged**, or **the projection cut in the middle** (all queries,
  then all gates). The second is the natural reading of a `(2dH, D)` matrix and gives
  the first four heads of the real model everything.
- **RoPE over the whole head.** `ops.partial_rope` rotates every channel unless it is
  given the factor; with it, 64 of 256 channels rotate.
- **RoPE settings as literals.** `1e7` and `0.25` are this checkpoint's; the second
  small model has $10^4$ and 0.5.
- **QK-norm weights as gains**, **the final norm as a gain**, **the block norms as
  gains**: three versions of one mistake. The final norm's stored weights average 3.3.
- **One added to the delta layer's own norm.** The opposite mistake, from the rule
  "this model stores offsets".
- **The default epsilon in the block norms.** `ops.rms_norm` defaults to $10^{-5}$,
  the model says $10^{-6}$. On the small model with $\epsilon = 10^{-6}$ and inputs of
  order one the logits stay within tolerance; on the real hidden states, whose mean
  square is near $10^{-3}$, 49 norms each off by a fraction of a percent move the
  logits by up to tenths. The second small model has $\epsilon = 10^{-2}$ so that this
  shows without the download.
- **The default epsilon in `q_norm` and `k_norm`.** Invisible on the real model: its
  queries and keys are far larger than $\sqrt{10^{-5}}$ before the norm, so both real
  cases pass. It is still a wrong port, and only a configuration with another epsilon
  shows it.
- **Scaled embeddings.** Gemma and the original Transformer multiply by
  $\sqrt{D}$; this model does not.
- **No causal mask** in the attention layers. The delta layers are causal by
  construction, so only six layers leak, and the last position is still right.
- **Layers in name order.** Sorting the names gives 0, 1, 10, 11, ..., 19, 2, 20, ...:
  correct for every model of at most ten layers.
- **Calling `ops.gated_delta_layer`.** Allowed in `hybrid-decoder-forward`, not here:
  this problem is the port without the kit's layer.

## A reference that is not one

When two libraries implement a model, either can serve as the reference, until they
disagree. Measured on this machine against `transformers` 5.23.0 in float32:

| `mlx-lm` 0.32.2 | eight prompts, every logit (the kit's measurement) | three prompts of 12 tokens, 64 columns |
|---|---|---|
| as it loads the checkpoint | 0.70, another argmax at 5 of 86 positions | 0.26, none of 36 |
| after `model.set_dtype(mx.float32)` | 0.17, 1 of 86 | 0.07, none of 36 |
| weights cast to float32 before its `sanitize` | $1.7 \cdot 10^{-4}$, none | not measured |

The cause is fact 3 meeting fact 11: `mlx-lm` adds the `1.0` to the offset norm weights
once, when it loads the file, and it adds it to bfloat16 numbers, so every gain is
rounded to eight bits of mantissa before any conversion to float32 can help (the final
norm's gains by up to 0.39 %). It is the mistake "compute in the storage dtype", made
once at load time. This problem allows an error of $10^{-3} + 10^{-3}\,|\text{logit}|$:
0.011 at a logit of 10 and 0.032 at the largest logits here (31). The smallest
difference in the first two rows, 0.07, is six times the allowance at a logit of 10
and twice the largest allowance, which is why the fixtures come from `transformers`. The
numbers do not say which library generates better text; they say that a tolerance is
only meaningful against a reference that computes what you compute.

## Complexity

Per token and layer about 22 million multiply-adds in a delta block and about 19
million in an attention block (plus $O(T)$ per token for the scores), 0.5 billion for
the 24 blocks; the head is $248{,}320 \times 1024 = 254$ million per position, as much
as twelve blocks. During generation only the last position needs the head.

## In the wild

`transformers` runs a multi-token call through a chunked form of the delta rule and a
one-token cached call through the recurrent form; on this model its own two paths
differ by up to $1.5 \cdot 10^{-4}$ in the logits (measured for this curriculum), the
same size as the difference between it and a from-scratch port. The checkpoint also
carries a 12-layer vision tower and a one-layer multi-token-prediction head (`mtp.*`),
which a text-only port never reads (`mtp-chain-draft` is about that head). The delta
layer is from "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv
2412.06464); the gated attention, the QK norm, and the storage of norm weights as
offsets are in the code, and no architecture paper for this model family was read for
this curriculum.
