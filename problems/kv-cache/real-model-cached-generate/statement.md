Generate text with a real model, IBM's Granite 4.0 350M, the way a runtime does it:
read the prompt in one forward pass, keep the attention keys and values of every
layer, and then push **one** token at a time through the 28 layers. You wrote the
forward pass in `real-model-logits`. Here it gets a memory, and the exercise is what
the memory must hold: keys that are already rotated, at the width of the 4 key/value
heads, with positions that continue where the cache ends.

![This problem is the whole loop, run on a real model.](assets/map.svg)

## Specification

```python
def prefill(weights, cfg, prompt_ids): ...  # returns (logits, cache)
def decode_step(weights, cfg, token, cache): ...  # returns (logits, cache)
def generate(weights, cfg, prompt_ids, max_new_tokens): ...  # returns a list of ints
```

`weights` and `cfg` are those of `real-model-logits`: a mapping from checkpoint names
to float32 tensors (shapes under [Shapes](#shapes); matrices are `(out, in)`, no
biases), and a [`RealConfig`](../../../python/src/learnai/kit/realmodel.py) with the
fields `n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `head_dim`, `d_ff`,
`vocab_size`, `norm_eps`, `rope_theta`, `attention_scale`, `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`, and `tied_embeddings`.

**The model.** For a sequence of tokens at positions $0, 1, 2, \dots$, with
$E$ = `model.embed_tokens.weight`:

1. $x = E[\text{tokens}] \cdot \texttt{embedding\_multiplier}$.
2. For layer $i = 0, \dots, \texttt{n\_layers} - 1$, with
   $r = \texttt{residual\_multiplier}$:
   $x \leftarrow x + r \cdot \operatorname{Attn}_i(\operatorname{RMSNorm}(x))$, then
   $x \leftarrow x + r \cdot \operatorname{MLP}_i(\operatorname{RMSNorm}(x))$.
   - RMSNorm is $u / \sqrt{\operatorname{mean}(u^2) + \texttt{norm\_eps}} \cdot g$,
     with the gains `input_layernorm.weight` and `post_attention_layernorm.weight`.
   - Attention has `n_heads` query heads and `n_kv_heads` key/value heads of width
     `head_dim`; query head $h$ uses key/value head
     $\lfloor h / (\texttt{n\_heads} / \texttt{n\_kv\_heads}) \rfloor$. Queries and
     keys (not values) are rotated with RoPE in the half-split layout: feature $j$ of a
     head pairs with feature $j + \texttt{head\_dim}/2$, the angle of pair $j$ at
     position $p$ is $p \cdot \texttt{rope\_theta}^{-2j/\texttt{head\_dim}}$, and
     $(a, b)$ becomes $(a \cos - b \sin,\; a \sin + b \cos)$. Scores are
     $q\,k^\top \cdot \texttt{attention\_scale}$; the token at position $p$ attends to
     positions $\le p$; softmax; weighted sum of the values; heads merged; `o_proj`.
   - The MLP computes $z = u\,W_{\text{in}}^\top$ with
     `shared_mlp.input_linear.weight`; the first `d_ff` features of $z$ are the gate
     and the rest are up; the output is
     $(\operatorname{silu}(\text{gate}) \odot \text{up})\,W_{\text{out}}^\top$ with
     `shared_mlp.output_linear.weight`.
3. $x \leftarrow \operatorname{RMSNorm}(x)$ with `model.norm.weight`; the logits are
   $x\,W_{\text{head}}^\top / \texttt{logits\_scaling}$, where $W_{\text{head}}$ is
   $E$ if `cfg.tied_embeddings` and `lm_head.weight` otherwise.

**`prefill(weights, cfg, prompt_ids)`** runs the model on the whole prompt, an integer
tensor of shape `(P,)`, $P \ge 1$, in one pass. It returns `(logits, cache)`: the
logits of the **last** position only, a float32 tensor of shape `(vocab_size,)`, and a
cache.

**`decode_step(weights, cfg, token, cache)`** takes one more token, a Python `int`,
and a cache that holds $n$ tokens. The token is at position $n$. It returns
`(logits, cache)`: the logits at position $n$, shape `(vocab_size,)`, equal to what
the model computes for the whole sequence of $n + 1$ tokens, and a cache that holds
$n + 1$ tokens.

**The cache** is yours to design; the tests only hand it back to `decode_step`. What
makes it a cache is the cost: `decode_step` runs one token through the layers and
reads everything about earlier tokens from the cache. That fixes what it holds: for
each layer, the keys and values of every token so far, at the width of the
`n_kv_heads` key/value heads (expand them to the query heads when you use them, not
when you store them). A key is rotated once, for its own position, when its token is
new.

**The cache is a value.** `decode_step` must not change the cache it is given. The
caller may continue from the same cache twice, with different tokens, and both
continuations must be right (this is what beam search, speculative decoding, and a
server that reuses a prompt's cache do).

**`generate(weights, cfg, prompt_ids, max_new_tokens)`** is greedy decoding:
`prefill`, then `max_new_tokens` times the token with the largest logit, each fed to
`decode_step` to get the logits for the next. It returns the generated tokens (not
the prompt) as a list of `max_new_tokens` Python ints. There is no stop token here.

Write the forward pass and the cache yourself, in float32. From the kit,
`ops.rms_norm` and `ops.softmax` are meant to be used. Not allowed: the kit's `RealLM`
and `RealModel`, the `transformers` and `mlx_lm` packages, and MLX's built-in rotary
embedding (`nn.RoPE`, `mx.fast.rope`).

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

The first public case is a three-layer model with random weights in Granite's layout
and Granite's constants: vocabulary 64, `d_model = 128`, 4 query heads on 2 key/value
heads of width 32, `d_ff = 96`, a tied head.

```python
generate(weights, cfg, [41], 8)  # [10, 10, 16, 16, 16, 16, 30, 38]
```

`prefill` reads the single token 41 at position 0 and returns logits whose largest
entry is at 10. Its cache holds one token: per layer, keys and values of shape
`(2, 1, 32)` if you store them as (KV heads, tokens, head width). `decode_step(…, 10,
cache)` computes the query, key, and value of token 10 at **position 1**, attends over
two keys, and returns logits with their maximum at 10 again; the cache now holds two
tokens. Seven steps after the prefill produce the eight tokens; the eighth token is
not fed, because nobody would read its logits.

On the real model, the 12 tokens of `"A B C D E F G H I J K L"` are continued with
`" M N O P Q R S T U V W X"`. A cache of 24 tokens holds, per layer, keys and values
of 4 KV heads times 24 tokens times 64 numbers: about 1.4 MB over the 28 layers. At
the width of the 16 query heads it would be four times that.

## Constraints

- Logits are compared with `transformers` (float32, computed without a cache) to a
  relative tolerance of `1e-3` and an absolute tolerance of `1e-3`: on the small
  models everywhere, on the real model at 64 fixed vocabulary columns. Tokens are
  compared exactly; every test prompt's greedy tokens win by a clear margin.
- $1 \le P \le 768$, and at most 12 tokens are generated. The cases without a
  download use two random three-layer models: the one of the example, and one with
  other multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and an untied head.
- **Speed.** After a `prefill` of 768 real tokens, four calls of `decode_step` are
  timed against the reference in the same framework, and may take at most **3 times**
  as long. A step that reads its cache costs one token through 28 layers; a step that
  runs the 769 tokens again costs about eight times that on a CPU, and more on a GPU.
- The time limit per case is 30 seconds (120 for the speed case). The first real-model
  case decodes all 352 M weights (1 to 2 seconds, once per run).

> [!NOTE]
> JAX compiles every operation once per combination of input shapes. A cache that
> grows by one slot per token changes the shapes of the attention at every step, and a
> first run then spends most of its time compiling. Keeping the cache in buffers
> whose capacity grows in larger steps (with a mask over the unused slots) avoids it.
> In PyTorch and MLX, appending is fine.

## Why this matters

This is the loop every LLM runtime spends its life in, and the split into `prefill`
and `decode_step` is the split behind the two numbers every benchmark reports: time
to first token is the prefill, tokens per second is the decode step. The cache is
what makes the second number independent of the length of the conversation. Its bugs
are quiet. A new token rotated for position 0 instead of position $n$, or cached keys
rotated a second time, still produce fluent text for a few tokens, because nearby
positions rotate almost alike; the text degrades with length, and the bug is filed as
"the model loses the thread in long chats". A cache updated in place works until the
first feature that needs to branch from it. And a runtime with no cache at all
produces exactly the right tokens, which is why the only test that finds it is a
clock.
