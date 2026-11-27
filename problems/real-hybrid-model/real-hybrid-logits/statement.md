Implement the forward pass of `Qwen/Qwen3.5-0.8B` from its checkpoint tensors to compute next-token logits matching Hugging Face `transformers`. Eighteen of the model's 24 layers use the Gated DeltaNet recurrent block from `real-gdn-block`, and the other six use grouped-query attention with a gated query projection, query-key RMSNorm, and partial RoPE.

## Specification

```python
def qwen_logits(weights, cfg, tokens): ...  # -> logits, (T, vocab_size)
```

- `weights`: a mapping from checkpoint tensor names to float32 tensors (either a `dict` or the kit's lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py), which supports the same indexing and iteration). Weight matrices have shape `(out, in)` and are applied as `u @ W.T`. There are no biases. Tensor names and shapes are listed in [Shapes](#shapes). The mapping may also include tensors unused by the text decoder (such as the vision tower and draft head, 168 tensors in the real checkpoint). Ignore them.
- `cfg`: a [`HybridConfig`](../../../python/src/learnai/kit/hybrid.py) object. This problem uses `cfg.layers` (a sequence of `"linear_attention"` or `"full_attention"` per layer, already resolved from `layer_types`), `cfg.n_layers`, `cfg.vocab_size`, `cfg.norm_eps`, `cfg.n_heads`, `cfg.n_kv_heads`, `cfg.head_dim`, `cfg.rope_base`, `cfg.partial_rotary_factor`, `cfg.n_key_heads`, `cfg.n_value_heads` (equal to `cfg.n_key_heads` in every test), `cfg.key_head_dim`, and `cfg.value_head_dim`.
- `tokens`: an integer tensor of shape `(T,)` containing token IDs for one sequence, with $T \ge 1$. Token $t$ is at position $t$.
- Return value: a float32 tensor of shape `(T, cfg.vocab_size)` where row $t$ holds the logits for the token following `tokens[t]`.

Most norms in this model use an offset RMSNorm where the stored weight $w$ is the gain minus one. With $\epsilon =$ `cfg.norm_eps` and the mean taken over the last axis, define

$$
\operatorname{Norm}(u; w) = \frac{u}{\sqrt{\operatorname{mean}(u^2) + \epsilon}} \odot (1 + w).
$$

Let `R = "model.language_model."` and, for layer index $i$, let `p = R + f"layers.{i}."`.

1. **Embed.** Set $x$ to the rows `tokens` of `R + "embed_tokens.weight"`. Do not scale the embeddings or add positional embeddings.
2. **Blocks.** For $i = 0, \dots,$ `cfg.n_layers - 1` in numeric order, update

$$
x \leftarrow x + \operatorname{Mixer}_i\big(\operatorname{Norm}(x;\, w_1)\big), \qquad
x \leftarrow x + \operatorname{MLP}\big(\operatorname{Norm}(x;\, w_2)\big),
$$

   where $w_1 =$ `p + "input_layernorm.weight"`, $w_2 =$ `p + "post_attention_layernorm.weight"`, and the MLP is `ops.swiglu(u, gate_proj, up_proj, down_proj)`.
3. **Final norm and head.** Apply $\operatorname{Norm}(x;\,$ `R + "norm.weight"`$)$ and multiply by the transpose of the embedding matrix. There is no separate `lm_head` tensor.

**The mixer of a `"linear_attention"` layer** is the recurrent mixer of `real-gdn-block` using the tensors under `p + "linear_attn."`, starting from a zero state:

1. Project the normalized hidden state with `in_proj_qkv`, run `ops.causal_conv1d` with `conv1d.weight`, and apply SiLU.
2. Split the result into queries $q$ and keys $k$ (head dimension $d_k =$ `cfg.key_head_dim`) and values $v$ (head dimension `cfg.value_head_dim`).
3. Apply `ops.l2norm` to $q$ and $k$ with $10^{-6}$ under the square root, then scale $q$ by $d_k^{-1/2}$.
4. Compute the per-head decay $g = \exp(-e^{\texttt{A\_log}} \operatorname{softplus}(a + \texttt{dt\_bias}))$ and update rate $\beta = \sigma(b)$ from the layer's $a$ and $b$ projections.
5. Starting from $S = 0$, update per head and token as $\bar S = g S$, $S = \bar S + \beta (v - \bar S k) k^\top$, and $y = S q$.
6. Normalize $y$ with RMSNorm using $\gamma =$ `norm.weight` **as stored** (a plain gain, not $1 + \gamma$) and $\epsilon =$ `cfg.norm_eps`, then gate elementwise by $\operatorname{silu}(z)$ using the layer's $z$ projection:

$$
\gamma \odot \frac{y}{\sqrt{\operatorname{mean}(y^2) + \epsilon}} \odot \operatorname{silu}(z).
$$

7. Project the gated output with `out_proj`.

**The mixer of a `"full_attention"` layer** uses the tensors under `p + "self_attn."`. Let $H =$ `cfg.n_heads`, $G =$ `cfg.n_kv_heads`, and $d =$ `cfg.head_dim`. For normalized input $u$:

1. Compute $u W_q^\top$, which has $2dH$ columns. Head $h$ owns columns $2dh$ through $2dh + 2d - 1$. Its first $d$ columns are its **query**, and its last $d$ columns are its **gate**. Compute $u W_k^\top$ and $u W_v^\top$, each with $dG$ columns, where head $j$ owns columns $dj$ through $dj + d - 1$.
2. Normalize queries and keys over their $d$ channels using the offset RMSNorm: $\operatorname{Norm}(q;\,$ `q_norm.weight`$)$ and $\operatorname{Norm}(k;\,$ `k_norm.weight`$)$. Do not normalize values.
3. Rotate queries and keys by position with partial RoPE. Let $r = \operatorname{int}(d \cdot$ `cfg.partial_rotary_factor`$)$ and $\theta =$ `cfg.rope_base`. For $0 \le i < r/2$, channel $i$ pairs with channel $i + r/2$. At position $t$, the pair $(a, b)$ becomes $(a \cos \phi - b \sin \phi,\ a \sin \phi + b \cos \phi)$ with $\phi = t\,\theta^{-2i/r}$. Channels $r$ through $d - 1$ pass through unchanged. You can compute this with `ops.partial_rope(z, positions, cfg.rope_base, cfg.partial_rotary_factor)` on tensors of shape `(heads, T, d)`.
4. Run causal grouped-query attention. Query head $h$ uses key and value head $\lfloor h / (H/G) \rfloor$, with scores $q k^\top / \sqrt{d}$. Token $t$ attends only to positions $s \le t$, followed by softmax over keys and a weighted sum of values. You can compute this with `ops.attention(q, k, v, mask)` using a boolean `(T, T)` mask that is `True` where attention is allowed.
5. Multiply the attended vectors of shape `(T, H, d)` elementwise by $\sigma(\text{gate})$ for the same token and head, merge to `(T, H * d)`, and multiply by $W_o^\top$.

![The output of the query projection and one rotated query head in Qwen3.5 0.8B. The gate half of each head only scales that head's output.](assets/query-head-layout.svg)

Implement the forward pass yourself. You may use `ops.rms_norm`, `ops.swiglu`, `ops.partial_rope`, `ops.attention`, `ops.l2norm`, `ops.softplus`, and `ops.causal_conv1d` from the kit. Do not call `ops.gated_delta_layer` (under any import path), any of the kit's delta-rule oracles or reference blocks and models (`gdn_step_ref`, `gdn_gates_ref`, `gdn_layer_ref`, `gated_attention_ref`, `hybrid_block_ref`, `hybrid_forward_ref`, `RealHybrid`, `RealHybridLM`), or the packages `transformers`, `safetensors`, and `mlx_lm`.

## Shapes

Let $D$ be the hidden dimension and $F$ be the MLP intermediate dimension.

| name | shape | dtype |
|---|---|---|
| `tokens` | `(T,)` | integer |
| result | `(T, vocab_size)` | float32 |
| `R + "embed_tokens.weight"` | `(vocab_size, D)` | float32 |
| `R + "norm.weight"`, `p + "input_layernorm.weight"`, `p + "post_attention_layernorm.weight"` | `(D,)` | float32 |
| `p + "mlp.gate_proj.weight"`, `up_proj.weight` / `down_proj.weight` | `(F, D)` / `(D, F)` | float32 |
| `p + "self_attn.q_proj.weight"` | `(2 * d * H, D)` | float32 |
| `p + "self_attn.k_proj.weight"`, `v_proj.weight` | `(d * G, D)` | float32 |
| `p + "self_attn.o_proj.weight"` | `(D, d * H)` | float32 |
| `p + "self_attn.q_norm.weight"`, `k_norm.weight` | `(d,)` | float32 |
| `p + "linear_attn.*"`, nine tensors | as in `real-gdn-block` | float32 |

In `Qwen/Qwen3.5-0.8B`, there are 24 layers with full attention at indices 3, 7, 11, 15, 19, and 23. The dimensions are $D = 1024$, $F = 3584$, and `vocab_size` $= 248{,}320$. Attention layers use $H = 8$, $G = 2$, and $d = 256$, so `q_proj` is `(4096, 1024)`, `k_proj` and `v_proj` are `(512, 1024)`, and `o_proj` is `(1024, 2048)`. Partial RoPE uses $\theta = 10^7$ and $r = 64$. Linear-attention layers use 16 delta heads with key and value head dimensions of 128. The norm epsilon is $\epsilon = 10^{-6}$.

## Example

The first public test case uses a four-layer model with random weights in the checkpoint layout and the real model's layer schedule (`cfg.layers` has three `"linear_attention"` layers followed by one `"full_attention"` layer). Its dimensions are $D = 32$, `vocab_size` $= 64$, $H = 4$, $G = 2$, and $d = 16$, making `q_proj` have shape `(128, 32)`. Partial RoPE uses $\theta = 10^7$ and factor 0.25, giving $r = 4$, so channels 0 and 1 pair with channels 2 and 3 while channels 4 through 15 are not rotated.

```python
tokens               # an integer tensor of shape (6,): [3, 1, 4, 1, 5, 9]
logits = qwen_logits(weights, cfg, tokens)
logits.shape         # (6, 64)
logits[5, :5]        # [-3.244581,  1.593436,  2.538823,  0.680665, -0.069021]
logits.argmax(-1)    # [49, 28, 60, 48, 22, 48]
```

In the attention layer, the query projection for the six tokens has shape `(6, 128)`. Reshaping to `(6, 4, 32)` and splitting along the last axis yields a query of shape `(6, 4, 16)` and a gate of shape `(6, 4, 16)`. Keys and values have shape `(6, 2, 16)`. Query heads 0 and 1 read key-value head 0, and query heads 2 and 3 read key-value head 1. The attended output of shape `(6, 4, 16)` is multiplied elementwise by $\sigma(\text{gate})$ and merged to `(6, 64)` before `o_proj`.

## Constraints

- Logits must be float32 and are compared against `transformers` 5.23 (`Qwen3_5ForCausalLM`, float32, single call) with a relative tolerance of `1e-3` and an absolute tolerance of `1e-3`. A correct float32 implementation differs from the fixtures by less than $10^{-4}$.
- Test cases that do not require a download run random models of width 32 on 6 tokens and on a single token. Two fixed models are judged against `transformers`: the example model above, and a six-layer model with layer order full, linear, linear, full, linear, full, $\theta = 10^4$, factor 0.5, and `cfg.norm_eps` $= 10^{-2}$. Additional randomized models vary the layer schedule, $\theta$, rotary factor, and `cfg.norm_eps` (between $10^{-4}$ and $3 \cdot 10^{-3}$) and are judged against the kit's NumPy forward pass on the same weights. Every RMSNorm (including `q_norm`, `k_norm`, and the delta layer's norm) must use `cfg.norm_eps`. Only `l2norm` uses a fixed $10^{-6}$.
- The real-model test cases run `Qwen/Qwen3.5-0.8B` (revision `2fc0636`) on two prompts of 12 token IDs. The judge checks the logits at 64 fixed vocabulary columns for every position, the 8 largest logits of the last position, and the argmax token of every position where the top two fixture logits differ by at least 0.01.
- $1 \le T \le 16$. This is a single-call forward pass with no cache, so layer states do not need to be returned.
- For the real model, weights are decoded from bfloat16 the first time each tensor is indexed (about 1 second total, once per run). The tied output head is a $248{,}320 \times 1024$ matrix, and a 12-token forward pass takes 0.05 to 0.5 seconds depending on the framework.

## Why this matters

Porting a new model architecture to an inference engine starts with a checkpoint of named tensors, a config file, and a reference implementation to match. Many critical details are not encoded in tensor shapes alone. A 4096-row query projection could be interpreted as 16 heads or as all queries followed by all gates instead of 8 heads of query-then-gate. Five types of norm weights are stored as gain minus one while a sixth is stored directly as a gain. Partial RoPE rotates only 64 of 256 channels. Every one of these mistakes loads and executes without raising an error, which is why engine ports are validated by numerical logit agreement against a reference rather than by inspecting generated text.
