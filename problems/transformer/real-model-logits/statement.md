Implement the full forward pass of IBM's Granite 4.0 350M language model from its
checkpoint tensors, mapping token IDs to vocabulary logits that match `transformers`
to within `1e-3`. Building on `decoder-forward` and `real-model-block`, you assemble
all layers together with Granite's embedding scaling, final normalization, and output
head while reading every constant from the model configuration.

![This problem is the whole forward pass of Granite 4.0 350M, from token ids to logits.](assets/map.svg)

## Specification

```python
def granite_logits(weights, cfg, tokens): ...
```

- `weights`: a mapping from checkpoint tensor names to float32 tensors (either a dict
  or the kit's lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py)). All
  linear weight matrices have shape `(out, in)`, and there are no bias vectors.
- `cfg`: a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py) containing the
  architecture hyperparameters read from `config.json`. The fields you need are listed
  below.

  | field | Granite 4.0 350M | meaning |
  |---|---|---|
  | `n_layers` | 28 | number of transformer blocks |
  | `d_model` | 1024 | width of the residual stream |
  | `n_heads`, `n_kv_heads` | 16, 4 | query heads, key/value heads |
  | `head_dim` | 64 | per-head dimension (`d_model // n_heads`) |
  | `d_ff` | 2048 | SwiGLU hidden width |
  | `norm_eps` | 1e-5 | epsilon for every RMSNorm |
  | `rope_theta` | 1e7 | RoPE base frequency |
  | `attention_scale` | 1/64 | multiplier on attention scores |
  | `embedding_multiplier` | 12.0 | multiplier on the embedding output |
  | `residual_multiplier` | 0.263 | multiplier on each block branch |
  | `logits_scaling` | 4.0 | divisor on the output logits |
  | `tied_embeddings` | `True` | whether the output head shares the embedding matrix |

- `tokens`: an integer tensor of shape `(B, T)`, where `B` is the batch size and `T`
  is the sequence length. Entries are token IDs in `[0, vocab_size)`, where
  `vocab_size` is the vocabulary size, and entry `t` in each row sits at sequence
  position `t`.
- Returns a float32 tensor of shape `(B, T, vocab_size)` containing the next-token
  logits at every position.

Let $E = \texttt{weights["model.embed\_tokens.weight"]}$. Compute the forward pass in
five stages:

**1. Embed and scale.** Look up the token embeddings and scale them by
`embedding_multiplier`:

$$
x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}.
$$

This produces a tensor of shape `(B, T, d_model)`. Do not add a positional embedding
or prepend a beginning-of-text token.

**2. Transformer blocks.** For $i = 0, 1, \dots, \texttt{n\_layers} - 1$ in numerical
order, apply the Granite block from `real-model-block` using weights prefixed with
`model.layers.{i}.`, token positions $0, \dots, T - 1$, and all constants from `cfg`:

$$
x \leftarrow x + r \cdot \operatorname{Attn}_i\big(\operatorname{RMSNorm}(x)\big), \qquad
x \leftarrow x + r \cdot \operatorname{MLP}_i\big(\operatorname{RMSNorm}(x)\big),
\qquad r = \texttt{residual\_multiplier}.
$$

- **RMSNorm**: For an input vector $u$ and gain vector $g$, compute
  $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$. The attention
  branch uses `input_layernorm.weight`, and the MLP branch uses
  `post_attention_layernorm.weight`.
- **Attention**: Project the normalized input into `n_heads` query heads ($q$) and
  `n_kv_heads` key ($k$) and value heads, each of width `head_dim`. Query head $h$
  shares key and value head $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$.
  Rotate queries and keys (not values) with RoPE in the half-split layout. Feature $j$
  of a head pairs with feature $j + \texttt{head\_dim} / 2$ at angle
  $\theta_{t,j} = t \cdot \texttt{rope\_theta}^{-2j / \texttt{head\_dim}}$ for position
  $t$, mapping $(a, b)$ to
  $(a \cos\theta_{t,j} - b \sin\theta_{t,j},\; a \sin\theta_{t,j} + b \cos\theta_{t,j})$.
  Scale attention scores as $q k^\top \cdot \texttt{attention\_scale}$, apply a causal
  mask so position $t$ attends only to positions $s \le t$, apply softmax, take the
  weighted sum of the values, merge the heads, and project with `o_proj.weight`.
- **MLP**: Project the normalized input using `shared_mlp.input_linear.weight` to
  width `2 * d_ff`. Split the last axis into the first `d_ff` features (`gate`) and
  the remaining `d_ff` features (`up`), then compute
  $(\operatorname{silu}(\text{gate}) \odot \text{up}) W_{\text{out}}^\top$ where
  $W_{\text{out}}$ is `shared_mlp.output_linear.weight`.

**3. Final norm.** Apply RMSNorm to $x$ using the gain `model.norm.weight`.

**4. Output head.** Set $W_{\text{head}}$ to $E$ when `cfg.tied_embeddings` is `True`,
and to `weights["lm_head.weight"]` when it is `False`. Always follow
`cfg.tied_embeddings` rather than inspecting the keys of `weights`. A tied checkpoint
normally omits `lm_head.weight`, and if a mapping includes one anyway, you must ignore
it.

**5. Logit scaling.** Project onto the vocabulary and divide by `logits_scaling`:

$$
\text{logits} = x W_{\text{head}}^\top / \texttt{logits\_scaling}.
$$

Run the entire forward pass yourself in float32. You may use `ops.rms_norm` and
`ops.softmax` from the kit. You may not use the kit's `RealLM` or `RealModel`, the
`transformers` package, or MLX's built-in rotary embeddings (`nn.RoPE`, `mx.fast.rope`).

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
| `lm_head.weight` | `(vocab_size, d_model)` (only when `tied_embeddings` is `False`) | float32 |
| result | `(B, T, vocab_size)` | float32 |

## Example

The first public test uses a three-layer model with random weights in Granite's layout
and constants: `vocab_size = 64`, `d_model = 128`, `n_heads = 4`, `n_kv_heads = 2`,
`head_dim = 32`, `d_ff = 96`, `tied_embeddings = True`, `embedding_multiplier = 12`,
`residual_multiplier = 0.263`, `attention_scale = 1/64`, `logits_scaling = 4`, and
`rope_theta = 1e7`.

```python
tokens = [[6, 62, 9, 50, 8, 6], [26, 25, 0, 37, 25, 43]]
logits = granite_logits(weights, cfg, tokens)
logits.shape  # (2, 6, 64)
logits[0, 5, :5]  # [-0.095271, 0.260368, 4.941798, 4.609704, -0.712952]
```

Throughout the blocks, the residual stream has shape `(2, 6, 128)`. Inside each block,
queries have shape `(2, 4, 6, 32)`, keys and values have shape `(2, 2, 6, 32)`, and
the fused MLP projection has shape `(2, 6, 192)`. Because `lm_head.weight` is absent
and embeddings are tied, the final step multiplies by the transposed embedding matrix
and divides by 4.

On the real Granite 4.0 350M checkpoint, the 12 tokens of `"The capital of France is
Paris. The capital of Japan is"` produce logits of shape `(1, 12, 100352)`. The last
position reaches its maximum value of `31.04` at token `27286` (`" Tokyo"`).

## Constraints

- Logits are compared against `transformers` in float32 with a relative tolerance of
  `1e-3` and an absolute tolerance of `1e-3`. Small models are checked across all
  entries. The real model is checked at 64 fixed vocabulary columns of every position
  and at the 8 largest logits of the last position.
- The argmax token must match the reference at every position where the two largest
  reference logits differ by more than `0.2`.
- The returned tensor must have dtype `float32`.
- `1 <= B <= 2` and `1 <= T <= 16`. Real-model prompts are 12 tokens long.
- The no-download tests use two random three-layer models. One matches the example
  above, and the other uses different multipliers, `rope_theta = 1e4`,
  `norm_eps = 1e-2`, and an untied head. `cfg.fused_mlp` is `True` in every test.
- The time limit per case is 20 seconds. The first real-model case decodes all 352M
  weights from bfloat16 to float32 once per run (1 to 2 seconds), and each forward
  pass then takes about 0.2 seconds on CPU.

## Why this matters

Reproducing reference logits from an existing checkpoint is the first verification
step for any new inference runtime. Until the raw logits match, downstream work on
sampling, KV caching, quantization, or evaluation is measuring a different model.

Most porting bugs come from small architectural conventions, such as Granite's
embedding multiplier, logit divisor, custom attention scale, or tied output head.
Missing one of these rarely crashes the program and still produces plausible text,
which makes the quality drop easy to misattribute to quantization or prompt wording.
Matching the reference logits to within `1e-3` confirms that you are running the exact
same model.
