Load IBM's Granite 4.2 3B from an 8-bit MLX checkpoint and run greedy generation with a KV cache and asynchronous evaluation. Keep weights packed end to end so the model runs in under 4 GiB instead of the 13.6 GiB required for `float32` weights.

## Specification

```python
def load_model(path): ...
def generate(model, prompt_ids, max_new_tokens, eos_id=None): ...
```

### `load_model(path)`

`path` is a directory containing two files: `config.json` and `model.safetensors`.

**`config.json`** has `model_type` `"granite"`. Calling `RealConfig.from_json(json.loads(text))` from the kit produces the same architecture fields as in `mlx-real-model-forward`, plus `quantization = (bits, group_size)`. For Granite 4.2 3B:

| field | Granite 4.2 3B | field | Granite 4.2 3B |
|---|---|---|---|
| `n_layers` | 40 | `attention_scale` | 1/64 |
| `d_model` | 2560 | `embedding_multiplier` | 1.0 |
| `n_heads`, `n_kv_heads` | 40, 8 | `residual_multiplier` | 1.0 |
| `head_dim` | 64 | `logits_scaling` | 1.0 |
| `d_ff` | 8192 | `rope_theta` | 1e7 |
| `vocab_size` | 100352 | `norm_eps` | 1e-5 |
| `quantization` | `(8, 64)` (`bits, group_size`) | `tied_embeddings` | `False` |

**`model.safetensors`** uses the layout written by `mlx_lm.convert -q`. Every linear layer and the embedding are stored as three tensors, and every RMSNorm gain is stored as one tensor. For a quantized weight matrix of logical shape `(out, in)` with bit width `bits` and group size $g = \texttt{group\_size}$:

| tensor | dtype | shape |
|---|---|---|
| `<layer>.weight` | `uint32` | `(out, in * bits / 32)` (packed codes, lowest bits first) |
| `<layer>.scales`, `<layer>.biases` | `bfloat16` | `(out, in / group_size)` |
| `<norm>.weight` | `bfloat16` | `(d_model,)` |

The dequantized matrix $\hat{W}$ has entries $\hat{W}_{o,i} = \text{code}_{o,i} \cdot \text{scales}_{o,\lfloor i/g \rfloor} + \text{biases}_{o,\lfloor i/g \rfloor}$ for output index $o$ and input index $i$. The checkpoint layers are:

```
model.embed_tokens                                  (vocab_size, d_model)
model.layers.{i}.input_layernorm                    norm
model.layers.{i}.self_attn.{q,k,v,o}_proj           as in mlx-real-model-forward
model.layers.{i}.post_attention_layernorm           norm
model.layers.{i}.mlp.gate_proj, .up_proj            (d_ff, d_model)
model.layers.{i}.mlp.down_proj                      (d_model, d_ff)
model.norm                                          norm
lm_head                                             (vocab_size, d_model)
```

Linear layers have no bias vectors, and the output head is always `lm_head` rather than a tied embedding.

`load_model` returns an `mlx.nn.Module` satisfying three requirements:

1. **Exact parameter tree.** `tree_flatten(model.parameters())` matches the checkpoint's tensor names exactly. Each quantized layer's `.weight` remains a `uint32` array of packed codes with its stored shape.
2. **Float32 floating parameters.** All floating parameters (`scales`, `biases`, and norm gains) are converted from `bfloat16` to `float32`, and all parameters are evaluated before `load_model` returns.
3. **Cached call interface.** The returned module supports two call forms for an integer array `tokens` of shape `(1, T)` with $T \ge 1$:

   ```python
   logits, cache = model(tokens)          # fresh forward pass
   logits, cache = model(tokens, cache)   # continue after cached tokens
   ```

   `logits` is `float32` with shape `(1, T, vocab_size)`. `cache` is an object of your design that stores keys and values for every layer, and calling `model(tokens, cache)` must not mutate the input `cache`. Given a `cache` holding $C$ prior tokens, the new tokens occupy positions $C, C + 1, \dots, C + T - 1$. Each new token attends to all $C$ cached tokens and to new tokens up to itself, producing the same logits those positions would get in a single pass over the full sequence.

**Forward pass.** The computation matches `mlx-real-model-forward` (embedding scaled by `embedding_multiplier`, pre-norm blocks with both branches scaled by `residual_multiplier`, half-split RoPE with base `rope_theta` on queries and keys, attention scores scaled by `attention_scale`, final RMSNorm, and logits divided by `logits_scaling`), with three differences:

- Every linear layer computes $y = x\,\hat{W}^\top$ directly from packed codes (`mx.quantized_matmul` or `nn.QuantizedLinear`), and the embedding looks up rows of the dequantized embedding matrix $\hat{W}_E$.
- For normalized input $u$, the MLP uses separate gate, up, and down projections: $\operatorname{down}\big(\operatorname{silu}(\operatorname{gate}(u)) \odot \operatorname{up}(u)\big)$.
- The output head is `lm_head`.

Read every constant from `config.json`, including `bits` and `group_size`.

**Graph requirements.** For an `n`-layer model, the lazy graph of `model(tokens)[0]` must contain `7n + 1` `QuantizedMatmul` operations (7 per layer plus `lm_head`), `n` `ScaledDotProductAttention` operations, zero `Matmul` operations, and at most one dequantization (for the embedding rows of `tokens`).

### `generate(model, prompt_ids, max_new_tokens, eos_id=None)`

Run greedy decoding with `model`, choosing the argmax of the last position's logits at each step.

- `prompt_ids`: list of $P \ge 1$ Python `int`s.
- `max_new_tokens`: integer $\ge 1$.
- `eos_id`: `int` or `None`. When generated, `eos_id` is appended as the last element and generation stops. An `eos_id` inside `prompt_ids` does not stop generation.
- Returns a list of $g$ Python `int`s with $1 \le g \le \texttt{max\_new\_tokens}$.

The judge wraps `model` to count calls and uses `EvalSpy` to inspect evaluations (as in `mlx-streaming-generate`), so call `model` only through its documented interface:

1. Call `model` once with the full prompt of shape `(1, P)` and no cache, then pass one token of shape `(1, 1)` plus `cache` on every subsequent step.
2. Do not call `mx.eval`. Schedule every step with `mx.async_eval` and read each generated token exactly once with `.item()`.
3. Schedule step $t + 1$ before reading token $t$ for every $t < \texttt{max\_new\_tokens} - 1$. Do not schedule any step after the token that reaches `max_new_tokens`.
4. Call `model` at most $g$ times when `max_new_tokens` ends generation, and at most $g + 1$ times when `eos_id` ends it early (the step already scheduled before reading `eos_id` is discarded).

**Forbidden.** Importing `mlx_lm` or `transformers`, or using `RealLM` or `RealModel` from the kit, is not allowed. Build the solution with `nn.QuantizedLinear`, `nn.QuantizedEmbedding`, `nn.quantize`, `mx.quantized_matmul`, and `RealConfig`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(1, T)` | integer |
| `logits` | `(1, T, vocab_size)` | `float32` |
| `prompt_ids` | list of $P$ ints | Python `int` |
| result of `generate` | list of $g$ ints | Python `int` |

## Example

The first public test uses a random three-layer model with the same layout: `vocab_size = 256`, `d_model = 128`, 2 query heads, 1 key/value head, `head_dim = 64`, `d_ff = 256`, the Granite 4.2 3B constants, and 8-bit quantization with group size 64. Its checkpoint has 76 tensors (3 for the embedding, 23 per layer for 7 quantized projections and 2 norm gains, 1 final norm gain, and 3 for `lm_head`).

```python
model = load_model("tiny-a")
# model.model.layers[0].self_attn.q_proj.weight: uint32, shape (128, 32)
# model.model.layers[0].self_attn.q_proj.scales: float32, shape (128, 2)

logits, cache = model(mx.array([[1, 17, 200, 45, 9, 130]]))
logits.shape  # (1, 6, 256)
logits[0, -1, :4]  # [2.323889, 3.797198, -0.293408, 0.827526]

generate(model, [1, 17, 200, 45, 9, 130], 12)
# [219, 154, 179, 99, 254, 82, 65, 155, 253, 192, 99, 209]
```

The forward graph contains $7 \cdot 3 + 1 = 22$ `QuantizedMatmul` operations. `generate` calls `model` 12 times (once with `(1, 6)` and 11 times with `(1, 1)`), with the evaluation sequence `async_eval, async_eval, read, async_eval, read, ..., read, read`.

On the real Granite 4.2 3B checkpoint, the 12 tokens of `"The capital of France is Paris. The capital of Japan is"` continue with `[27286, 13, 578, 6864, 315, 16327, 374, 62224]` (`" Tokyo. The capital of Brazil is Bras"`).

## Constraints

- Logits are compared against `mlx-lm` running the same checkpoint with `float32` scales, biases, and norm gains, with relative and absolute tolerance `1e-3`. All logits are checked on the small models. On the real model, the next-token logits of each prompt are checked at 64 fixed vocabulary columns and at the 8 largest logits. (By default `mlx-lm` runs the checkpoint in `bfloat16`, which shifts logits by up to about `0.4`.)
- Generated tokens must match exactly. On the real model, every tested greedy step has a margin of at least `0.5` between the top two logits.
- Evaluation order and model call counts are checked strictly.
- **Peak memory.** On the Apple GPU, loading the real model and generating 8 tokens must peak at no more than **6 GiB** above baseline. The reference peaks at 3.97 GiB (3.84 GiB of parameters), whereas dequantized weights require 13.6 GiB in `float32` and 6.8 GiB in 16 bits.
- The non-download tests use two random three-layer models: the example model above, and a second model with different multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and 4-bit quantization in groups of 32. Real prompts are 12 to 22 tokens long and generate at most 8 tokens.
- The time limit is 20 seconds per case on the small models and 120 seconds on the real model, where the reference takes about 0.2 seconds to load from page cache and 0.2 seconds to generate.

## Why this matters

Local inference on a Mac works because weights stay quantized from disk to matrix multiplication. They are read as packed words, stored as packed words, and multiplied directly in `mx.quantized_matmul`.

On unified memory, where the GPU shares system RAM, dequantizing weights at load time or inside the forward pass does not raise an error. The model returns the right tokens while running three times slower and using four times the memory, quietly pushing other processes into swap. Writing `load_model` and `generate` here mirrors `mlx_lm.load` and `mlx_lm.generate_step`: build the quantized tree from `config.json`, load strictly, and decode with a KV cache and `mx.async_eval`.
