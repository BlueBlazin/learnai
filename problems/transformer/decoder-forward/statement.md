Implement the whole forward pass of a modern decoder-only transformer: token ids in,
next-token logits out, from a dictionary of checkpoint weights. Every building block
is one you have already written and is available from the kit; the work is assembling
them in the right order, with the right tensor in each slot, for a layout in which
almost every mistake still produces logits of the right shape.

![This problem is the whole forward pass, from token ids to next-token probabilities.](assets/map.svg)

## Specification

```python
def forward(params, cfg, tokens): ...
```

The model is the layout shared by Llama, Qwen, Mistral, and Granite: token embedding
without a position embedding, `cfg.n_layers` pre-norm blocks with rotary positions and
grouped-query attention, a final norm, and a linear output head.

`cfg` is a [`DecoderConfig`](../../../python/src/learnai/kit/decoder.py) with the
fields `vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_mlp`,
`rope_base`, `norm_eps`, `tie_embeddings`, and the derived
`head_dim = d_model // n_heads`. `params` maps the names in the table under
[Shapes](#shapes) to tensors. All linear weights are in `(out, in)` layout, and there
are no biases. Write $D$ for `d_model`, $H$ for `n_heads`, $H_{kv}$ for `n_kv_heads`,
and $d$ for `head_dim`.

**1. Embed.** $x = E[\text{tokens}]$ with $E$ = `model.embed_tokens.weight`: shape
`(B, T, D)`. The embeddings are not scaled.

**2. Blocks.** For layer $i = 0, \dots, $ `n_layers` $- 1$, in order, with the weights
named `model.layers.{i}.*`:

$$
x \leftarrow x + \operatorname{Attn}_i\big(\operatorname{RMSNorm}(x;\, g^{(i)}_1)\big), \qquad
x \leftarrow x + \operatorname{MLP}_i\big(\operatorname{RMSNorm}(x;\, g^{(i)}_2)\big),
$$

where $g_1$ is `input_layernorm.weight`, $g_2$ is `post_attention_layernorm.weight`,
and every RMSNorm in the model uses `eps = cfg.norm_eps`.

$\operatorname{Attn}_i(u)$, for a normalized input $u$ of shape `(B, T, D)`:

1. Project: $q = u\,W_q^\top$ of shape `(B, T, H * d)`, and $k = u\,W_k^\top$,
   $v = u\,W_v^\top$ of shape `(B, T, H_kv * d)`.
2. Split heads: reshape the last axis into `(heads, d)` and move the head axis forward.
   $q$ becomes `(B, H, T, d)`; $k$ and $v$ become `(B, H_kv, T, d)`. Head $h$ owns
   features $h d, \dots, (h + 1) d - 1$ of the projection.
3. Rotate $q$ and $k$, and not $v$, with RoPE: within each head, the feature pair
   $(2j, 2j + 1)$ of the vector at position $t$ is rotated counterclockwise by the
   angle $t \cdot \texttt{rope\_base}^{-2j/d}$. Positions are $t = 0, \dots, T - 1$ in
   every row. This is `ops.rope(x, positions, cfg.rope_base)`.
4. Attend causally with grouped-query attention: query head $h$ uses key/value head
   $\lfloor h / (H / H_{kv}) \rfloor$, the scores are divided by $\sqrt{d}$, and query
   position $t$ attends to key positions $s \le t$ only. This is
   `ops.attention(q, k, v, mask)` with a boolean mask that is `True` where $s \le t$;
   the result has shape `(B, H, T, d)`.
5. Merge heads back to `(B, T, H * d)` (the inverse of step 2) and project with
   $W_o^\top$.

$\operatorname{MLP}_i(u)$ is the SwiGLU MLP, `ops.swiglu(u, W_gate, W_up, W_down)`.

**3. Final norm.** $x \leftarrow \operatorname{RMSNorm}(x;\, g)$ with $g$ =
`model.norm.weight`.

**4. Head.** $\text{logits} = x\,W_{\text{head}}^\top$, where $W_{\text{head}}$ is

- `model.embed_tokens.weight` if `cfg.tie_embeddings` is `True`. In that case `params`
  has **no** key `lm_head.weight`, exactly like a checkpoint with tied embeddings;
- `lm_head.weight` if `cfg.tie_embeddings` is `False`.

Arguments and result:

- `params`: a dict from names to float32 tensors.
- `cfg`: the `DecoderConfig` the weights were made for.
- `tokens`: integer tensor of shape `(B, T)`, ids in `[0, vocab_size)`.
- Returns the logits: float32, shape `(B, T, vocab_size)`. Row `[b, t]` scores the
  token that follows `tokens[b, : t + 1]`.

Use the kit's [`ops`](../../../python/src/learnai/kit/ops/__init__.py) for the
blocks: `ops.rms_norm`, `ops.rope`, `ops.attention` (which does the grouped-query
pairing when `k` and `v` have fewer heads than `q`), and `ops.swiglu`. Do not call the
kit's own forward pass, `decoder_forward_ref` or `decoder_block_ref`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(B, T)` | integer |
| `model.embed_tokens.weight` | `(vocab_size, d_model)` | float32 |
| `model.layers.{i}.input_layernorm.weight` | `(d_model,)` | float32 |
| `model.layers.{i}.self_attn.q_proj.weight` | `(n_heads * head_dim, d_model)` | float32 |
| `model.layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` | float32 |
| `model.layers.{i}.self_attn.o_proj.weight` | `(d_model, n_heads * head_dim)` | float32 |
| `model.layers.{i}.post_attention_layernorm.weight` | `(d_model,)` | float32 |
| `model.layers.{i}.mlp.gate_proj.weight`, `up_proj.weight` | `(d_mlp, d_model)` | float32 |
| `model.layers.{i}.mlp.down_proj.weight` | `(d_model, d_mlp)` | float32 |
| `model.norm.weight` | `(d_model,)` | float32 |
| `lm_head.weight` (only when untied) | `(vocab_size, d_model)` | float32 |
| result | `(B, T, vocab_size)` | float32 |

## Example

A one-layer model of width 4 with two query heads sharing one key/value head, and the
kit's seeded weights:

```python
from learnai.kit import DecoderConfig, decoder_params

cfg = DecoderConfig(vocab_size=5, d_model=4, n_layers=1, n_heads=2, n_kv_heads=1, d_mlp=6)
params = decoder_params(cfg, seed=0)  # 11 tensors; no "lm_head.weight": the head is tied
tokens = [[1, 3, 0]]

forward(params, cfg, tokens)
# [[[ 0.142855,  1.460640, -0.848208, -0.733341,  0.875444],
#   [-0.519588,  0.535796,  2.398094,  6.208618,  1.158294],
#   [ 0.527882, -0.490735, -2.421456, -5.390296, -1.365599]]]
```

Here `head_dim` is 2, so `q_proj.weight` is `(4, 4)` and `k_proj.weight` and
`v_proj.weight` are `(2, 4)`; each head holds a single RoPE pair, rotated by $t$
radians at position $t$.

The model is causal, which you can see by changing the last token only:

```python
forward(params, cfg, [[1, 3, 4]])
# [[[ 0.142855,  1.460640, -0.848208, -0.733341,  0.875444],
#   [-0.519588,  0.535796,  2.398094,  6.208618,  1.158294],
#   [ 0.037625, -0.767143,  0.937001, -1.124330,  0.166404]]]
```

The first two rows are identical, because positions 0 and 1 cannot see position 2.

## Constraints

- Results are compared with the kit's oracle to a relative tolerance of `1e-3` and an
  absolute tolerance of `1e-4` (the pass is a deep stack; the logits have a standard
  deviation of about 2).
- $1 \le B \le 4$, $2 \le T \le 16$, `n_layers` $\le 4$, `d_model` $\le 32$,
  `vocab_size` $\le 64$. `n_heads` is a multiple of `n_kv_heads`, and `head_dim` is
  even.
- `cfg.norm_eps` and `cfg.rope_base` are not always the defaults; read them from `cfg`.
- Do not call `decoder_forward_ref` or `decoder_block_ref`.

## Why this matters

This function is what an inference engine *is*: everything else in serving (the KV
cache, batching, quantization, speculative decoding) is a way of computing these same
logits faster or cheaper, and each of them is tested by comparing against this plain
pass. It is also the code you write when you port a model to a new runtime, and the
port is where the conventions bite. A checkpoint lists tensors, not wiring: it does not
say that values are left unrotated, that the stream gets one more norm before the
head, or that a missing `lm_head.weight` means "reuse the embedding". Get one of these
wrong and nothing crashes; the model just becomes somewhat, or completely, worse.
