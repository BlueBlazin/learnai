Implement cached autoregressive text generation for IBM's Granite 4.0 350M model. You will process the prompt in one forward pass (`prefill`), store each layer's attention keys and values in a KV cache, and then advance one token at a time (`decode_step`). Each cached key is already rotated for its position and stored at the width of the key/value heads, so new tokens continue seamlessly from where the cache ends.

![This problem is the whole loop, run on a real model.](assets/map.svg)

## Specification

```python
def prefill(weights, cfg, prompt_ids): ...  # returns (logits, cache)
def decode_step(weights, cfg, token, cache): ...  # returns (logits, cache)
def generate(weights, cfg, prompt_ids, max_new_tokens): ...  # returns a list of ints
```

`weights` and `cfg` follow the same format as `real-model-logits`. `weights` maps checkpoint tensor names to float32 tensors (linear weights have shape `(out, in)` with no biases; see [Shapes](#shapes)). `cfg` is a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py) with fields `n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `head_dim`, `d_ff`, `vocab_size`, `norm_eps`, `rope_theta`, `attention_scale`, `embedding_multiplier`, `residual_multiplier`, `logits_scaling`, and `tied_embeddings`.

**The model.** For a sequence of token IDs at positions $0, 1, 2, \dots$, let $E$ be `model.embed_tokens.weight`:

1. **Embedding.** Compute $x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}$.
2. **Transformer layers.** For each layer $i = 0, \dots, \texttt{n\_layers} - 1$, let $r = \texttt{residual\_multiplier}$. Update $x \leftarrow x + r \cdot \operatorname{Attn}_i(\operatorname{RMSNorm}(x))$, then $x \leftarrow x + r \cdot \operatorname{MLP}_i(\operatorname{RMSNorm}(x))$.
   - **RMSNorm.** For an input vector $u$ and gain vector $g$ (`input_layernorm.weight` or `post_attention_layernorm.weight`), compute $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$.
   - **Attention.** Project the normalized input into `n_heads` query heads and `n_kv_heads` key/value heads, each of width `head_dim`. Under grouped-query attention, query head $h$ attends to key/value head $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$. Rotate queries and keys (not values) with RoPE in the half-split layout: feature $j$ pairs with feature $j + \texttt{head\_dim}/2$, using angle $\theta_{p,j} = p \cdot \texttt{rope\_theta}^{-2j/\texttt{head\_dim}}$ at position $p$, so $(a, b)$ becomes $(a \cos \theta_{p,j} - b \sin \theta_{p,j},\; a \sin \theta_{p,j} + b \cos \theta_{p,j})$. Compute attention scores $q\,k^\top \cdot \texttt{attention\_scale}$, apply a causal mask so position $p$ attends only to positions $\le p$, take the softmax, form the weighted sum of values, merge the heads back to `d_model`, and project with `o_proj`.
   - **MLP.** Compute $z = u\,W_{\text{in}}^\top$ using `shared_mlp.input_linear.weight`. Split $z$ into `gate` (the first `d_ff` features) and `up` (the remaining `d_ff` features), then return $(\operatorname{silu}(\text{gate}) \odot \text{up})\,W_{\text{out}}^\top$ using `shared_mlp.output_linear.weight`.
3. **Output head.** Apply RMSNorm with `model.norm.weight` to get the normalized final states, then compute logits $x\,W_{\text{head}}^\top / \texttt{logits\_scaling}$. Here $W_{\text{head}}$ is $E$ when `cfg.tied_embeddings` is `True`, and `lm_head.weight` otherwise.

**`prefill(weights, cfg, prompt_ids)`** runs the model on the entire prompt `prompt_ids` (an integer tensor of shape `(P,)` with $P \ge 1$) in one pass. It returns `(logits, cache)`, where `logits` holds the float32 logits for the **last** prompt position only, shape `(vocab_size,)`.

**`decode_step(weights, cfg, token, cache)`** takes a single next token (a Python `int`) and a cache holding $n$ prior tokens. The new token sits at position $n$. It returns `(logits, cache)`, where `logits` has shape `(vocab_size,)` and matches the logits at position $n$ of a full forward pass over all $n + 1$ tokens, and the returned cache holds $n + 1$ tokens.

**Cache design.** You choose the representation of `cache`; the tests treat it as opaque and only pass it back to `decode_step`. To achieve cached decode speed, `decode_step` must run only the single new token through the transformer layers and read all past keys and values from the cache. For each layer, store keys and values at the width of the `n_kv_heads` key/value heads (expand to `n_heads` when computing attention, not in storage). Rotate each key once for its own position when its token is processed.

**Immutability.** `decode_step` must not mutate the cache passed to it. A caller may branch from the same cache twice with different tokens, and both continuations must produce exact results (as required by beam search, speculative decoding, and shared-prefix serving).

**`generate(weights, cfg, prompt_ids, max_new_tokens)`** performs greedy decoding. Call `prefill`, then repeat `max_new_tokens` times: pick the token ID with the largest logit and pass it to `decode_step` to obtain the next logits (skipping `decode_step` after picking the final token). Return the generated tokens (excluding the prompt) as a list of `max_new_tokens` Python ints. There is no stop token.

Write the forward pass and cache yourself in float32. You may use `ops.rms_norm` and `ops.softmax` from the kit. You may not use the kit's `RealLM` or `RealModel`, the `transformers` or `mlx_lm` packages, or MLX's built-in rotary embeddings (`nn.RoPE`, `mx.fast.rope`).

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt_ids` | `(P,)` | integer |
| `token` | Python `int` | |
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
| logits of `prefill` and `decode_step` | `(vocab_size,)` | float32 |
| result of `generate` | a list of `max_new_tokens` Python ints | |

## Example

The first public test uses a random three-layer model with Granite's layout and constants: `vocab_size = 64`, `d_model = 128`, 4 query heads, 2 key/value heads of width 32, `d_ff = 96`, and a tied output head.

```python
generate(weights, cfg, [41], 8)  # [10, 10, 16, 16, 16, 16, 30, 38]
```

`prefill` processes token 41 at position 0 and returns logits whose argmax is 10. Its cache holds one token per layer (for example, keys and values of shape `(2, 1, 32)`). Next, `decode_step(..., 10, cache)` computes the query, key, and value of token 10 at **position 1**, attends over the two cached keys, and returns logits whose argmax is 10 again, along with a cache of two tokens. Seven `decode_step` calls after `prefill` yield all eight tokens. The eighth token is not passed back into `decode_step` because its logits are not needed.

On the real Granite 4.0 350M model, the 12 tokens of `"A B C D E F G H I J K L"` continue with `" M N O P Q R S T U V W X"`. A 24-token cache stores, per layer, keys and values for 4 KV heads × 24 tokens × 64 floats, totaling about 1.4 MB across all 28 layers. Storing all 16 expanded query heads instead would use four times as much memory.

## Constraints

- Logits are compared against `transformers` (float32, without a cache) with relative tolerance `1e-3` and absolute tolerance `1e-3`. All vocabulary entries are checked on the small models, and 64 fixed vocabulary columns are checked on the real model. Generated token IDs are compared exactly; every test prompt's greedy tokens win by a clear margin.
- $1 \le P \le 768$, and at most 12 tokens are generated. The download-free tests use two random three-layer models: the example model above, and a second model with different multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and an untied output head.
- **Speed.** After a `prefill` of 768 real tokens, four calls to `decode_step` are timed against the reference in the same framework and must take at most **3 times** as long. A cached step runs one token through 28 layers, whereas recomputing all 769 tokens takes roughly eight times as long on CPU (and even more on GPU).
- The time limit is 30 seconds per case (120 seconds for the speed case). The first real-model test decodes all 352 M weights once per run (about 1 to 2 seconds).

> [!NOTE]
> JAX compiles every operation once per combination of input shapes. If a cache grows by one slot per token, the attention shapes change on every step, and the first run spends most of its time compiling. Storing the cache in buffers whose capacity grows in larger chunks (with a mask over unused slots) avoids repeated recompilation. In PyTorch and MLX, appending directly is fine.

## Why this matters

Every LLM serving engine is built around this exact split between `prefill` and `decode_step`. Those two functions directly determine the two headline serving metrics. Time to first token (TTFT) is `prefill`, and tokens per second is `decode_step`. The KV cache is what keeps decode cost nearly independent of conversation length.

Cache bugs are notoriously silent. Rotating a new token at position 0 instead of position $n$, or rotating cached keys a second time, still produces fluent text for several tokens because nearby RoPE angles are close. Quality degrades gradually with length, looking like a model issue rather than a cache bug. Mutating a cache in place passes single-sequence tests and breaks only when a caller branches from an earlier state. Skipping the cache altogether produces identical tokens and can only be caught by timing.
