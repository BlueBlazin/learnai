Load a real 3-billion-parameter model, IBM's Granite 4.2 3B as an 8-bit MLX
checkpoint, and generate text from it, in under 4 GiB where its float32 weights alone
would take 13.6. Everything you built in this track meets here: a module tree that
spells the checkpoint's names, linear layers that multiply straight from packed
codes, the fused kernels, a KV cache, and a generation loop that schedules the next
step before it reads the current token. The one thing that must never happen is the
float weight matrix.

## Specification

```python
def load_model(path): ...
def generate(model, prompt_ids, max_new_tokens, eos_id=None): ...
```

### `load_model(path)`

`path` is a directory with two files.

**`config.json`**, of `model_type` `"granite"`. `RealConfig.from_json(json.loads(text))`
from the kit turns it into the fields of `mlx-real-model-forward`; for this model:

| field | Granite 4.2 3B | field | Granite 4.2 3B |
|---|---|---|---|
| `n_layers` | 40 | `attention_scale` | 1/64 |
| `d_model` | 2560 | `embedding_multiplier` | 1.0 |
| `n_heads`, `n_kv_heads` | 40, 8 | `residual_multiplier` | 1.0 |
| `head_dim` | 64 | `logits_scaling` | 1.0 |
| `d_ff` | 8192 | `rope_theta` | 1e7 |
| `vocab_size` | 100352 | `norm_eps` | 1e-5 |
| `quantization` | `(8, 64)`: `(bits, group_size)` | `tied_embeddings` | `False` |

**`model.safetensors`**, in the layout `mlx_lm.convert -q` writes. Every linear layer
and the embedding is stored as three tensors, and every RMSNorm gain as one:

| tensor | dtype | shape |
|---|---|---|
| `<layer>.weight` | uint32 | `(out, in * bits / 32)`: the packed codes, lowest bits first |
| `<layer>.scales`, `<layer>.biases` | bfloat16 | `(out, in / group_size)` |
| `<norm>.weight` | bfloat16 | `(d_model,)` |

with $\hat W_{o,i} = \text{code}_{o,i} \cdot \text{scales}_{o,\lfloor i/g \rfloor} +
\text{biases}_{o,\lfloor i/g \rfloor}$ the matrix a layer stands for. The layers are

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

There are no biases of the linear layers themselves, and the head is always
`lm_head`, never the embedding.

`load_model` returns an `mlx.nn.Module` with these properties:

1. **Its parameters are the checkpoint's tensors**: `tree_flatten(model.parameters())`
   has exactly the names of the file. Each `.weight` of a quantized layer is still the
   uint32 array of packed codes with its stored shape.
2. **Every floating parameter is float32** (scales, biases, norm gains are converted
   from bfloat16), and all parameters are evaluated.
3. **It is called** as

   ```python
   logits, cache = model(tokens)  # tokens: integer array of shape (1, T), T >= 1
   logits, cache = model(tokens, cache)  # the tokens that follow the cached ones
   ```

   `logits` is float32 of shape `(1, T, vocab_size)`. `cache` is an object of your
   own design holding the keys and values of every layer; the model never modifies
   one it was given. A call with a cache of $C$ tokens places its tokens at positions
   $C, C + 1, \dots$, lets each attend to all cached tokens and to the new tokens up
   to itself, and returns exactly the logits those positions get in one pass over the
   whole sequence.

**The forward pass** is the one of `mlx-real-model-forward` (embedding times
`embedding_multiplier`; pre-norm blocks with both branches times
`residual_multiplier`; RoPE in the half-split layout with base `rope_theta` on
queries and keys; scores times `attention_scale`; final norm; head; division by
`logits_scaling`), with three differences:

- every linear layer computes $y = x\,\hat W^\top$ **from the packed codes**
  (`mx.quantized_matmul`, or `nn.QuantizedLinear`, which calls it), and the embedding
  returns rows of $\hat W_E$;
- the MLP has separate matrices:
  $\operatorname{down}\big(\operatorname{silu}(\operatorname{gate}(u)) \odot
  \operatorname{up}(u)\big)$;
- the head is `lm_head`.

Every constant comes from the config, including `bits` and `group_size`.

**The graph of a forward pass.** For an `n`-layer model, the lazy graph of
`model(tokens)[0]` must contain `7n + 1` `QuantizedMatmul` operations (seven per layer
and the head), `n` `ScaledDotProductAttention` operations, no `Matmul`, and at most
one dequantization (the embedding rows of the given tokens).

### `generate(model, prompt_ids, max_new_tokens, eos_id=None)`

Greedy decoding with the model above: the next token is the argmax of the logits of
the last position.

- `prompt_ids`: a list of $P \ge 1$ ints. `max_new_tokens`: an int $\ge 1$.
- `eos_id`: an int or `None`. Generation stops after generating `eos_id`, which is
  the last element of the result. An `eos_id` in the prompt stops nothing.
- Returns the generated tokens, a list of $1 \le g \le$ `max_new_tokens` Python ints.

The judge hands `generate` a wrapper that counts the calls, so use `model` only as
documented, and it observes the evaluations (`EvalSpy`), as in
`mlx-streaming-generate`:

1. The first call passes the whole prompt, shape `(1, P)`; every later call passes
   one token, shape `(1, 1)`, and the cache.
2. `mx.eval` is not called. Each step is scheduled with `mx.async_eval`, and each
   token is read exactly once, with `.item()`.
3. **Step $t + 1$ is scheduled before token $t$ is read**, for every $t$ below
   `max_new_tokens - 1`. No step is built after the token that reaches the limit.
4. The model is called at most $g$ times when the limit ends the generation, and at
   most $g + 1$ times when an EOS ends it before the limit: the step that was
   scheduled before the EOS was read is abandoned.

**Not allowed:** `mlx_lm`, `transformers`, and the kit's `RealLM` and `RealModel`.
`nn.QuantizedLinear`, `nn.QuantizedEmbedding`, `nn.quantize`, `mx.quantized_matmul`,
and `RealConfig` are what the solution is made of.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(1, T)` | integer |
| `logits` | `(1, T, vocab_size)` | float32 |
| `prompt_ids` | list of $P$ ints | Python `int` |
| result of `generate` | list of $g$ ints | Python `int` |

## Example

The first public case is a three-layer model in the same layout: vocabulary 256,
`d_model = 128`, 2 query heads on 1 key/value head of width 64, `d_ff = 256`, the
constants of the table above, 8 bits in groups of 64. Its checkpoint has 76 tensors:
three for the embedding, 23 per layer (seven layers of three tensors and two norm
gains), one final norm gain, and three for the head.

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

The graph of the forward pass holds $7 \cdot 3 + 1 = 22$ `QuantizedMatmul`
operations. `generate` calls the model 12 times: once with shape `(1, 6)`, then 11
times with `(1, 1)`; the evaluations read
`async_eval, async_eval, read, async_eval, read, ..., read, read`.

On the real model, the 12 tokens of `"The capital of France is Paris. The capital of
Japan is"` continue with `[27286, 13, 578, 6864, 315, 16327, 374, 62224]`:
`" Tokyo. The capital of Brazil is Bras"`.

## Constraints

- Logits are compared with `mlx-lm` running the same checkpoint with float32 scales,
  biases, and gains, to a relative and an absolute tolerance of `1e-3`: on the small
  models everywhere; on the real model, the next-token logits of each prompt at 64
  fixed vocabulary columns and at the 8 largest. (`mlx-lm` by default runs the
  checkpoint in bfloat16, which moves logits by up to about 0.4.)
- Generated tokens are compared exactly. On the real model every stored greedy step
  has a margin of at least 0.5 between the two largest logits.
- The evaluation rules and call counts are exact.
- **Memory**, on the Apple GPU: loading the real model and generating 8 tokens peaks
  at no more than **6 GiB** above what was in use before. The reference peaks at
  3.97 GiB (3.84 GiB of parameters); the dequantized weights are 13.6 GiB in float32
  and 6.8 GiB in 16 bits.
- The cases without a download use two random three-layer models: the one of the
  example, and one with other multipliers, `rope_theta = 1e4`, `norm_eps = 1e-2`, and
  4 bits in groups of 32. Real prompts are 12 to 22 tokens long and generate at most
  8 tokens.
- The time limit is 20 seconds per case on the small models and 120 seconds on the
  real one, where the reference needs about 0.2 seconds to load (from the page
  cache) and 0.2 seconds to generate.

## Why this matters

This is what running a model locally means on a Mac. The reason a 3B model fits a
laptop comfortably and a 30B model fits a workstation at all is that the weights stay
quantized end to end: read from disk as packed words, held as packed words, and
multiplied as packed words. Every layer you write that turns them back into floats,
at load or on each call, gives the saving back, and on a machine where the GPU's
memory is the system's memory there is no error to tell you: the model answers
correctly, three times slower and at four times the footprint, until something else
on the machine is swapped out. `mlx_lm.load` is the function you are writing, and
`mlx_lm.generate_step` is the loop: build the tree from `config.json`, quantize the
modules the config says are quantized, load strictly, then generate with a cache and
`mx.async_eval`.
