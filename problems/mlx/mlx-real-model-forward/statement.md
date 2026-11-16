Build IBM's Granite 4.0 350M as an `mlx.nn` module: a class whose parameter tree
matches the original checkpoint name for name, so that MLX's own `load_weights`
fills it from the 705 MB safetensors file, and whose forward pass is made of MLX's
fused kernels and reproduces the logits of `transformers`. In `real-model-logits` you
wrote this model as one function over a dict of tensors. Here it becomes what an MLX
runtime actually ships: a module that loads, converts between dtypes, and runs on the
GPU.

## Specification

```python
class Granite(nn.Module):
    def __init__(self, cfg): ...
    def __call__(self, tokens): ...
```

`cfg` is a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py), the
architecture read from `config.json`:

| field | Granite 4.0 350M | meaning |
|---|---|---|
| `n_layers` | 28 | number of blocks |
| `d_model` | 1024 | width of the residual stream |
| `n_heads`, `n_kv_heads` | 16, 4 | query heads, key/value heads |
| `head_dim` | 64 | `d_model // n_heads` |
| `d_ff` | 2048 | SwiGLU width |
| `vocab_size` | 100352 | |
| `norm_eps` | 1e-5 | epsilon of every RMSNorm |
| `rope_theta` | 1e7 | RoPE base |
| `attention_scale` | 1/64 | factor on the attention scores |
| `embedding_multiplier` | 12.0 | factor on the embedding output |
| `residual_multiplier` | 0.263 | factor on each branch of a block |
| `logits_scaling` | 4.0 | divisor of the logits |
| `tied_embeddings` | `True` | the head is the embedding matrix |

**The parameter tree.** After `Granite(cfg)`, the parameters of the module, named as
`mlx.utils.tree_flatten(model.parameters())` names them, are exactly the tensors of
the checkpoint, with the shapes under [Shapes](#shapes): no parameter more, none
fewer, none renamed. In particular a tied model has **no** `lm_head.weight`, and an
untied one has it. The initial values do not matter. The judge then runs

```python
model = Granite(cfg)
model.load_weights(path, strict=True)  # mlx.nn.Module.load_weights, not overridden
model.set_dtype(dtype)  # float32, float16, or bfloat16
logits = model(tokens)
```

**The forward pass** is the one of `real-model-logits`, with every constant taken
from `cfg`:

1. $x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}$, with $E$ the
   embedding matrix. No position embedding.
2. For each layer $i$ in order, with $r = \texttt{residual\_multiplier}$:

   $$
   x \leftarrow x + r \cdot \operatorname{Attn}_i\big(\operatorname{RMSNorm}(x)\big), \qquad
   x \leftarrow x + r \cdot \operatorname{MLP}_i\big(\operatorname{RMSNorm}(x)\big).
   $$

   - RMSNorm: $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$.
   - Attention: `n_heads` query heads and `n_kv_heads` key/value heads of width
     `head_dim`; query head $h$ uses key/value head
     $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$. Queries and
     keys are rotated with RoPE in the half-split layout (feature $j$ pairs with
     $j + \texttt{head\_dim}/2$, base `rope_theta`), token $t$ at position $t$.
     Scores are $q\,k^\top \cdot \texttt{attention\_scale}$; position $t$ attends to
     positions $s \le t$; softmax; weighted sum of the values; heads merged;
     `o_proj`.
   - MLP: $z = u\,W_{\text{in}}^\top$ has $2 \cdot \texttt{d\_ff}$ features; the
     first `d_ff` are the gate and the rest are up; the output is
     $(\operatorname{silu}(\text{gate}) \odot \text{up})\,W_{\text{out}}^\top$.
3. $x \leftarrow \operatorname{RMSNorm}(x)$ with `model.norm.weight`.
4. $\text{logits} = x\,W_{\text{head}}^\top / \texttt{logits\_scaling}$, where
   $W_{\text{head}}$ is $E$ if `cfg.tied_embeddings` and `lm_head.weight` otherwise.

- `tokens`: integer array of shape `(B, T)`.
- Returns the logits, shape `(B, T, vocab_size)`, **in the dtype of the parameters**:
  a model converted with `set_dtype(mx.float16)` computes and returns float16.

**The rules of the exercise.**

- Use MLX's fused kernels. The judge reads the lazy graph of one forward pass of an
  `n`-layer model: it must contain `n` `ScaledDotProductAttention` operations
  (`mx.fast.scaled_dot_product_attention`), `2n` `RoPE` operations (`nn.RoPE` or
  `mx.fast.rope`), and `2n + 1` `RMSNorm` operations (`nn.RMSNorm` or
  `mx.fast.rms_norm`).
- Write the module yourself: `mlx_lm`, `transformers`, and the kit's `RealLM` and
  `RealModel` are not allowed. `nn.Linear`, `nn.Embedding`, `nn.RMSNorm`, `nn.RoPE`,
  and `nn.silu` are what the module is made of.

> [!NOTE]
> `mx.fast.scaled_dot_product_attention` has a fused kernel only for head dimensions
> 64, 96, 128, 192, 256, and 512 (MLX 0.32.4); for any other it expands into ordinary
> operations when it is called. Every model in the tests has `head_dim = 64`.

## Shapes

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
| `lm_head.weight` | `(vocab_size, d_model)`; only if `tied_embeddings` is `False` |

| name | shape | dtype |
|---|---|---|
| `tokens` | `(B, T)` | integer |
| result | `(B, T, vocab_size)` | the dtype of the parameters |

The checkpoint stores every tensor as bfloat16; there are no biases.

## Example

The first public case is a three-layer model with random weights in Granite's layout
and Granite's constants: vocabulary 64, `d_model = 256`, 4 query heads on 2 key/value
heads of width 64, `d_ff = 96`, tied head. Its checkpoint holds 26 tensors:
`model.embed_tokens.weight`, eight per layer, and `model.norm.weight`.

```python
model = Granite(cfg)
model.load_weights("tiny-a.safetensors", strict=True)
model.set_dtype(mx.float32)

tokens = mx.array([[51, 6, 45, 48, 31, 34], [28, 17, 35, 52, 18, 35]])
logits = model(tokens)
logits.shape  # (2, 6, 64)
logits[0, 5, :5]  # [-3.320329, -0.238478, -5.581573, 13.10126, -0.75092]
```

The graph of that call holds 3 `ScaledDotProductAttention`, 6 `RoPE`, and 7 `RMSNorm`
operations. Inside a block the queries are `(2, 4, 6, 64)` and the keys and values
`(2, 2, 6, 64)`; the attention kernel takes them as they are.

On the real model, the 12 tokens of `"The capital of France is Paris. The capital of
Japan is"` give logits of shape `(1, 12, 100352)` whose last row has its maximum,
31.04, at token 27286, `" Tokyo"`.

## Constraints

- $1 \le B \le 2$, $1 \le T \le 12$.
- In float32 the logits are compared with `transformers` (float32) to a relative and
  an absolute tolerance of `1e-3`: on the small models everywhere; on the real model
  at 64 fixed vocabulary columns of every position and at the 8 largest logits of
  the last position, with the argmax token required to be equal wherever the two
  largest reference logits are more than 0.2 apart.
- In 16 bits the absolute tolerance is 0.25 for float16 (small models only) and 1.0
  for bfloat16 (small models and the real one, with the argmax compared where the
  margin exceeds 2). The result must have the model's dtype.
- The cases without a download use two random three-layer models: the one of the
  example, and one with other multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and
  an untied head.
- The time limit per case is 30 seconds; loading and converting the real checkpoint
  takes about 0.1 seconds on this machine and a 12-token forward pass 15 ms.

## Why this matters

Every model `mlx-lm` runs is a file like the one you are about to write: a module
tree whose attribute names are the checkpoint's tensor names, so that loading is
`load_weights` and nothing else, plus a forward pass assembled from `mx.fast`
kernels. Porting a new architecture to Apple silicon means writing exactly this, and
the checks are the ones here. A strict load proves the tree: one misnamed submodule
and 226 tensors have nowhere to go. Fixture logits prove the arithmetic: a missing
multiplier or the textbook attention scale still produces fluent text, from a
different model. And the dtype check proves the module can run in 16 bits, where one
float32 constant in the forward pass silently doubles the memory traffic of every
layer after it.
