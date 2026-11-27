Implement IBM's Granite 4.0 350M as an `mlx.nn.Module` whose parameter tree matches the original safetensors checkpoint name for name so that MLX's `load_weights` loads it directly. The forward pass must use MLX's fused kernels, support dtype conversion via `set_dtype`, and reproduce the logits of `transformers`.

## Specification

```python
class Granite(nn.Module):
    def __init__(self, cfg): ...
    def __call__(self, tokens): ...
```

`cfg` is a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py) holding the architecture fields read from `config.json`:

| field | Granite 4.0 350M | meaning |
|---|---|---|
| `n_layers` | 28 | number of transformer blocks |
| `d_model` | 1024 | width of the residual stream |
| `n_heads`, `n_kv_heads` | 16, 4 | query heads and key/value heads |
| `head_dim` | 64 | per-head width (`d_model // n_heads`) |
| `d_ff` | 2048 | SwiGLU hidden width |
| `vocab_size` | 100352 | vocabulary size |
| `norm_eps` | 1e-5 | epsilon for every RMSNorm |
| `rope_theta` | 1e7 | RoPE frequency base |
| `attention_scale` | 1/64 | multiplier on attention scores |
| `embedding_multiplier` | 12.0 | multiplier on embedding outputs |
| `residual_multiplier` | 0.263 | multiplier on each block branch |
| `logits_scaling` | 4.0 | divisor applied to final logits |
| `tied_embeddings` | `True` | whether the output head shares the embedding matrix |

**Parameter tree.** After `Granite(cfg)` is constructed, the parameter names returned by `mlx.utils.tree_flatten(model.parameters())` must match the checkpoint's tensor names and shapes in [Shapes](#shapes) with no extra, missing, or renamed parameters. A tied model (`cfg.tied_embeddings == True`) must not have `lm_head.weight`, and an untied model must have it. Initial parameter values do not matter. The judge loads and runs the model as follows:

```python
model = Granite(cfg)
model.load_weights(path, strict=True)  # mlx.nn.Module.load_weights, not overridden
model.set_dtype(dtype)  # float32, float16, or bfloat16
logits = model(tokens)
```

**Forward pass.** Given an integer array `tokens` of shape `(B, T)` (batch size `B`, sequence length `T`), compute logits of shape `(B, T, vocab_size)` using every constant from `cfg`:

1. Look up token embeddings from the embedding matrix $E$ (`model.embed_tokens.weight`) and scale them:

   $$x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}.$$

   There is no learned position embedding.
2. Pass $x$ through layers $i = 0, \dots, \texttt{n\_layers} - 1$ in order. Letting $r = \texttt{residual\_multiplier}$, each block updates $x$ with pre-norm residual connections:

   $$
   x \leftarrow x + r \cdot \operatorname{Attn}_i\big(\operatorname{RMSNorm}(x)\big), \qquad
   x \leftarrow x + r \cdot \operatorname{MLP}_i\big(\operatorname{RMSNorm}(x)\big).
   $$

   - **RMSNorm:** For an input vector $u$ along the last axis and learned gain weight $g$, compute $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$.
   - **Attention:** Project the normalized input into `n_heads` query heads and `n_kv_heads` key and value heads, each of width `head_dim`. Query head $h$ attends to key/value head $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$. Rotate queries and keys with RoPE in the half-split layout (feature $j$ pairs with $j + \texttt{head\_dim}/2$, base `rope_theta`), with token $t$ at position $t$. Scale the query-key dot products by `attention_scale` and apply a causal mask so position $t$ attends only to positions $s \le t$. Take the softmax, compute the weighted sum of values, merge the heads, and apply `o_proj`.
   - **MLP:** Project the normalized input $u$ with $W_{\text{in}}$ (`input_linear.weight`) to obtain $z = u\,W_{\text{in}}^\top$ with $2 \cdot \texttt{d\_ff}$ features. Split $z$ along the last axis so the first `d_ff` features are `gate` and the remaining `d_ff` features are `up`, then compute $(\operatorname{silu}(\text{gate}) \odot \text{up})\,W_{\text{out}}^\top$ using $W_{\text{out}}$ (`output_linear.weight`).
3. Apply the final normalization $x \leftarrow \operatorname{RMSNorm}(x)$ using `model.norm.weight`.
4. Project to vocabulary logits and divide by `logits_scaling`:

   $$\text{logits} = x\,W_{\text{head}}^\top / \texttt{logits\_scaling},$$

   where $W_{\text{head}}$ is $E$ when `cfg.tied_embeddings` is `True` and `lm_head.weight` otherwise.

The returned `logits` must have the same dtype as the model's parameters. For example, a model converted with `set_dtype(mx.float16)` must compute and return `float16`.

**Rules.**

- **Fused kernels required.** For a model with `n` layers, the judge inspects the lazy computation graph of one forward pass. It must contain `n` `ScaledDotProductAttention` operations (`mx.fast.scaled_dot_product_attention`), `2n` `RoPE` operations (`nn.RoPE` or `mx.fast.rope`), and `2n + 1` `RMSNorm` operations (`nn.RMSNorm` or `mx.fast.rms_norm`).
- **Build the module yourself.** Importing `mlx_lm` or `transformers`, or using `RealLM` or `RealModel` from the kit, is forbidden. Build the module from `nn.Linear`, `nn.Embedding`, `nn.RMSNorm`, `nn.RoPE`, and `nn.silu`.

> [!NOTE]
> `mx.fast.scaled_dot_product_attention` has a fused kernel only for head dimensions 64, 96, 128, 192, 256, and 512 (MLX 0.32.4). For any other head dimension it expands into ordinary operations when called. Every model in the tests has `head_dim = 64`.

## Shapes

The checkpoint stores every parameter in `bfloat16`, and no layer has a bias term.

| parameter | shape |
|---|---|
| `model.embed_tokens.weight` | `(vocab_size, d_model)` |
| `model.layers.{i}.input_layernorm.weight` | `(d_model,)` |
| `model.layers.{i}.self_attn.q_proj.weight` | `(n_heads * head_dim, d_model)` |
| `model.layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` |
| `model.layers.{i}.self_attn.o_proj.weight` | `(d_model, n_heads * head_dim)` |
| `model.layers.{i}.post_attention_layernorm.weight` | `(d_model,)` |
| `model.layers.{i}.shared_mlp.input_linear.weight` | `(2 * d_ff, d_model)` |
| `model.layers.{i}.shared_mlp.output_linear.weight` | `(d_model, d_ff)` |
| `model.norm.weight` | `(d_model,)` |
| `lm_head.weight` | `(vocab_size, d_model)` (only if `tied_embeddings` is `False`) |

| name | shape | dtype |
|---|---|---|
| `tokens` | `(B, T)` | integer |
| result | `(B, T, vocab_size)` | same dtype as model parameters |

## Example

The first public test uses a random three-layer model with Granite's layout and constants: `vocab_size = 64`, `d_model = 256`, 4 query heads, 2 key/value heads, `head_dim = 64`, `d_ff = 96`, and a tied head. Its checkpoint contains 26 tensors (`model.embed_tokens.weight`, 8 tensors per layer, and `model.norm.weight`).

```python
model = Granite(cfg)
model.load_weights("tiny-a.safetensors", strict=True)
model.set_dtype(mx.float32)

tokens = mx.array([[51, 6, 45, 48, 31, 34], [28, 17, 35, 52, 18, 35]])
logits = model(tokens)
logits.shape  # (2, 6, 64)
logits[0, 5, :5]  # [-3.320329, -0.238478, -5.581573, 13.10126, -0.75092]
```

The lazy graph of that call contains 3 `ScaledDotProductAttention`, 6 `RoPE`, and 7 `RMSNorm` operations. Inside each block, the query tensor has shape `(2, 4, 6, 64)` and the key and value tensors have shape `(2, 2, 6, 64)`, which the attention kernel accepts directly without repeating heads.

On the real Granite 4.0 350M checkpoint, the 12 tokens of `"The capital of France is Paris. The capital of Japan is"` produce logits of shape `(1, 12, 100352)`. The last row reaches its maximum of `31.04` at token `27286` (`" Tokyo"`).

## Constraints

- $1 \le B \le 2$, $1 \le T \le 12$.
- In `float32`, logits are compared against `transformers` (`float32`) with relative and absolute tolerance `1e-3`. On the small models all logits are checked. On the real model, the check covers 64 fixed vocabulary columns at every position and the 8 largest logits at the last position, and the argmax token must match wherever the top two reference logits differ by more than `0.2`.
- In 16-bit precision, the absolute tolerance is `0.25` for `float16` (small models only) and `1.0` for `bfloat16` (small models and the real model, with the argmax checked wherever the margin exceeds `2`). The output dtype must match the model's dtype.
- The non-download tests use two random three-layer models: the example model above, and a second model with different multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and an untied head.
- The time limit is 30 seconds per case. Loading and converting the real checkpoint takes about 0.1 seconds on the reference machine, and a 12-token forward pass takes about 15 ms.

## Why this matters

Every model in `mlx-lm` follows this exact pattern. Its module attributes mirror the checkpoint's tensor names so `load_weights` works with no custom remapping, and its forward pass composes `mx.fast` kernels for Apple silicon.

Each check in this problem catches a common silent failure when porting a model. Strict loading verifies the parameter tree before any math runs, because one misnamed submodule leaves 226 tensors unmatched. Fixture logits verify scaling constants and RoPE settings, because a missing multiplier or wrong attention scale can still generate fluent text from the wrong distribution. Checking the output dtype verifies 16-bit execution, where a single `float32` array constant silently upcasts activations and doubles memory traffic for every layer that follows.
