Run the whole forward pass of a real model, IBM's Granite 4.0 350M, from its
checkpoint tensors: token ids in, 100,352 logits per position out, matching
`transformers` to three decimal places. You have written every piece: the generic
decoder in `decoder-forward`, and this model's block in `real-model-block`. What is
left is the frame around the 28 blocks, where this model departs from the textbook in
three more places, and the discipline to take every number from the config.

## Specification

```python
def granite_logits(weights, cfg, tokens): ...
```

- `weights`: a mapping from checkpoint names to float32 tensors (a dict, or the kit's
  lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py)), with the names
  and shapes under [Shapes](#shapes). Matrices are `(out, in)`; there are no biases.
- `cfg`: a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py), the
  architecture read from `config.json`. The fields you need:

  | field | Granite 4.0 350M | meaning |
  |---|---|---|
  | `n_layers` | 28 | number of blocks |
  | `d_model` | 1024 | width of the residual stream |
  | `n_heads`, `n_kv_heads` | 16, 4 | query heads, key/value heads |
  | `head_dim` | 64 | `d_model // n_heads` |
  | `d_ff` | 2048 | SwiGLU width $F$ |
  | `norm_eps` | 1e-5 | epsilon of every RMSNorm |
  | `rope_theta` | 1e7 | RoPE base |
  | `attention_scale` | 1/64 | factor on the attention scores |
  | `embedding_multiplier` | 12.0 | factor on the embedding output |
  | `residual_multiplier` | 0.263 | factor on each branch of a block |
  | `logits_scaling` | 4.0 | divisor of the logits |
  | `tied_embeddings` | `True` | the head is the embedding matrix |

- `tokens`: integer tensor of shape `(B, T)`, ids in `[0, vocab_size)`. Token $t$ of
  every row is at position $t$.
- Returns the logits: float32, shape `(B, T, vocab_size)`.

The forward pass, with $E$ = `model.embed_tokens.weight`:

**1. Embed and scale.** $x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}$,
shape `(B, T, d_model)`. Nothing else is added: no position embedding, no
beginning-of-text token.

**2. Blocks.** For $i = 0, 1, \dots, \texttt{n\_layers} - 1$, **in this order**, apply
the block of `real-model-block` with the weights `model.layers.{i}.*`, positions
$0, \dots, T - 1$, and every constant taken from `cfg`:

$$
x \leftarrow x + r \cdot \operatorname{Attn}_i\big(\operatorname{RMSNorm}(x)\big), \qquad
x \leftarrow x + r \cdot \operatorname{MLP}_i\big(\operatorname{RMSNorm}(x)\big),
\qquad r = \texttt{residual\_multiplier}.
$$

- RMSNorm: $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$, with
  the gains `input_layernorm.weight` and `post_attention_layernorm.weight`.
- Attention: `n_heads` query heads and `n_kv_heads` key/value heads of width
  `head_dim`; query head $h$ uses key/value head
  $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$. Queries and keys
  (not values) are rotated with RoPE in the half-split layout: feature $j$ of a head
  pairs with feature $j + \texttt{head\_dim}/2$, the angle at position $t$ is
  $t \cdot \texttt{rope\_theta}^{-2j/\texttt{head\_dim}}$, and $(a, b)$ becomes
  $(a \cos - b \sin,\; a \sin + b \cos)$. Scores are
  $q\,k^\top \cdot \texttt{attention\_scale}$; position $t$ attends to positions
  $s \le t$; softmax; weighted sum of the values; heads merged; `o_proj`.
- MLP: $z = u\,W_{\text{in}}^\top$ with `shared_mlp.input_linear.weight` of shape
  `(2 * d_ff, d_model)`; the first `d_ff` features of $z$ are the gate and the rest
  are up; the output is $(\operatorname{silu}(\text{gate}) \odot \text{up})\,
  W_{\text{out}}^\top$ with `shared_mlp.output_linear.weight`.

**3. Final norm.** $x \leftarrow \operatorname{RMSNorm}(x)$ with the gain
`model.norm.weight`.

**4. Head.** $W_{\text{head}}$ is $E$ itself if `cfg.tied_embeddings` is `True`, and
`lm_head.weight` otherwise. The config decides, not the contents of `weights`: the
checkpoint of a tied model has no `lm_head.weight`, and if a mapping for a tied model
contains one anyway, it is not the head.

**5. Scale.** $\text{logits} = x\,W_{\text{head}}^\top \,/\, \texttt{logits\_scaling}$.

Run the forward pass yourself, in float32. From the kit, `ops.rms_norm` and
`ops.softmax` are meant to be used. Not allowed: the kit's `RealLM` and `RealModel`,
the `transformers` package, and MLX's built-in rotary embedding (`nn.RoPE`,
`mx.fast.rope`).

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
| `model.layers.{i}.shared_mlp.input_linear.weight` | `(2 * d_ff, d_model)` | float32 |
| `model.layers.{i}.shared_mlp.output_linear.weight` | `(d_model, d_ff)` | float32 |
| `model.norm.weight` | `(d_model,)` | float32 |
| `lm_head.weight` | `(vocab_size, d_model)`; only if `tied_embeddings` is `False` | float32 |
| result | `(B, T, vocab_size)` | float32 |

## Example

The first public case is a three-layer model with random weights in Granite's layout
and Granite's constants: vocabulary 64, `d_model = 128`, 4 query heads on 2 key/value
heads of width 32, `d_ff = 96`, tied head, `embedding_multiplier = 12`,
`residual_multiplier = 0.263`, `attention_scale = 1/64`, `logits_scaling = 4`,
`rope_theta = 1e7`.

```python
tokens = [[6, 62, 9, 50, 8, 6], [26, 25, 0, 37, 25, 43]]
logits = granite_logits(weights, cfg, tokens)
logits.shape  # (2, 6, 64)
logits[0, 5, :5]  # [-0.095271, 0.260368, 4.941798, 4.609704, -0.712952]
```

The stream is `(2, 6, 128)` throughout; inside a block the queries are
`(2, 4, 6, 32)` and the keys and values `(2, 2, 6, 32)`; the fused MLP projection is
`(2, 6, 192)`. There is no `lm_head.weight` in `weights`: the last step multiplies by
the transposed embedding matrix and divides by 4.

On the real model, the 12 tokens of `"The capital of France is Paris. The capital of
Japan is"` give logits of shape `(1, 12, 100352)` whose last row has its maximum,
31.04, at token 27286, `" Tokyo"`.

## Constraints

- Logits are compared with `transformers` (float32) to a relative tolerance of `1e-3`
  and an absolute tolerance of `1e-3`: on the small models everywhere, on the real
  model at 64 fixed vocabulary columns of every position and at the 8 largest logits
  of the last position. The argmax token must be equal at every position where the
  two largest reference logits are more than 0.2 apart. The result must be float32.
- $1 \le B \le 2$ and $1 \le T \le 16$. Real prompts are 12 tokens long.
- The cases without a download use two random three-layer models: the one of the
  example, and one with other multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and
  an untied head. `cfg.fused_mlp` is `True` in every test.
- The time limit per case is 20 seconds. The first real-model case decodes all 352 M
  weights from bfloat16 to float32 (1 to 2 seconds, once per run); a forward pass then
  takes about 0.2 seconds on a CPU.

## Why this matters

This is the test every new runtime has to pass before anything else about it matters:
load someone else's checkpoint and reproduce their logits. Until the numbers match,
sampling, caching, quantization, and benchmarks are all measurements of a different
model. The distance between "runs" and "matches" is a handful of conventions (here: a
multiplier on the embeddings, a divisor on the logits, a scale on the attention
scores, a head that is not in the file), each of which leaves the output looking
like a language model's. A port with one of them missing still completes sentences.
It is just worse, in a way that is easy to blame on quantization or on the prompt.
Matching to 1e-3 is how you know it is the same model.
