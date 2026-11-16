Build a runner for a quantized decoder in four stages: turn a float checkpoint into
packed 4-bit or 8-bit codes with per-group scales and biases, run the forward pass
straight from those codes, generate through a KV cache, and choose a mixed-precision
recipe that fits a size budget without moving the model's predictions too far. Each
stage is one earlier problem with its edges exposed; the capstone is that the float
weights are gone after stage 1, so every later stage has to live on what the
checkpoint really holds.

## Overview

```python
def quantize_checkpoint(params, bits, group_size): ...  # stage 1 -> checkpoint
def forward_q(ckpt, cfg, tokens, group_size): ...  # stage 2 -> logits
def step_q(ckpt, cfg, tokens, cache, group_size): ...  # stage 3 -> (logits, cache)
def generate_q(ckpt, cfg, prompt, max_new_tokens, group_size, cache=None): ...  # stage 3 -> tokens
def bits_per_weight(ckpt, group_size): ...  # stage 4 -> float
def quantize_mixed(params, group_size, max_bpw): ...  # stage 4 -> checkpoint
```

![The runner. Stages 1 and 4 turn float parameters into a checkpoint of packed codes, per-group scales and biases, and float gains; the float matrices are not stored. A step (stages 2 and 3) reads only the checkpoint: it dequantizes the embedding rows it looks up, multiplies by codes in every linear layer, and keeps rotated keys and values in the KV cache.](assets/runner.svg)

The cases are grouped by stage, and each stage's cases use only what that stage and
the ones before it specify. The tests of stages 2 and 3 pass in checkpoints that the
judge built from the rules of stage 1, so a forward pass can only use what the
format holds; stage 4 runs your own checkpoint through your own forward pass.

**The model** is the kit decoder of `decoder-forward`, described by a
[`DecoderConfig`](../../../python/src/learnai/kit/decoder.py) `cfg` (fields
`vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_mlp`, `rope_base`,
`norm_eps`, `tie_embeddings`, and the derived `head_dim = d_model // n_heads`).
`params` maps checkpoint names to float32 tensors. Linear weights are `(out, in)`
matrices without biases:

| name | shape |
|---|---|
| `model.embed_tokens.weight` | `(vocab_size, d_model)` |
| `model.layers.{i}.input_layernorm.weight`, `post_attention_layernorm.weight` | `(d_model,)` |
| `model.layers.{i}.self_attn.q_proj.weight` | `(n_heads * head_dim, d_model)` |
| `model.layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` |
| `model.layers.{i}.self_attn.o_proj.weight` | `(d_model, n_heads * head_dim)` |
| `model.layers.{i}.mlp.gate_proj.weight`, `up_proj.weight` | `(d_mlp, d_model)` |
| `model.layers.{i}.mlp.down_proj.weight` | `(d_model, d_mlp)` |
| `model.norm.weight` | `(d_model,)` |
| `lm_head.weight`, only when `cfg.tie_embeddings` is `False` | `(vocab_size, d_model)` |

You may use the kit's [`ops`](../../../python/src/learnai/kit/ops/__init__.py):
`ops.rms_norm`, `ops.rope`, and `ops.attention`. Do not call the kit's forward pass
(`decoder_forward_ref`, `decoder_block_ref`).

## Stage 1: the quantized checkpoint

`quantize_checkpoint(params, bits, group_size)` returns a new dict, the
**checkpoint**. `bits` is 4 or 8. Every tensor of `params` with two dimensions is a
matrix and is quantized; every other tensor (the norm gains) is stored unchanged
under its own name.

**Quantizing a matrix** `w` of shape `(out, in)` is `groupwise-4bit`. Each row is
cut into `in / group_size` consecutive groups along the input axis; group $g$ of row
$i$ holds the inputs $j$ with $g \cdot \text{group\_size} \le j < (g + 1) \cdot
\text{group\_size}$. With $L = 2^{\text{bits}} - 1$:

$$
\text{bias}_{i,g} = \min_{j \in g} w_{i,j},
\qquad
\text{scale}_{i,g} = \begin{cases}
(\max_{j \in g} w_{i,j} - \min_{j \in g} w_{i,j}) / L & \text{if the maximum exceeds the minimum}, \\
1 & \text{if every element of the group is equal},
\end{cases}
$$

$$
\text{code}_{i,j} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{i,j} - \text{bias}_{i,g}}{\text{scale}_{i,g}}\Big),\ 0,\ L\Big),
\qquad
\hat w_{i,j} = \text{code}_{i,j} \cdot \text{scale}_{i,g} + \text{bias}_{i,g} .
$$

$\operatorname{round}$ rounds to the nearest integer, ties to even (the `round` of
PyTorch, JAX, and MLX). $\hat w$ is the weight the codes stand for; it is never
stored.

**Packing.** The codes of a row are stored as bytes (uint8):

- `bits = 8`: one code per byte. The packed row has `in` bytes.
- `bits = 4`: two codes per byte, the first in the low half. Byte $b$ of a row is
  $\text{code}_{2b} + 16 \cdot \text{code}_{2b+1}$, and the packed row has `in / 2`
  bytes. This is the nibble order of `bit-packing`.

**Layout.** For a matrix named `P.weight` the checkpoint has exactly three entries,
and the float matrix is not one of them:

| key | shape | dtype | content |
|---|---|---|---|
| `P.weight` | `(out, in * bits / 8)` | uint8 | the packed codes |
| `P.scales` | `(out, in / group_size)` | float32 | $\text{scale}_{i,g}$ |
| `P.biases` | `(out, in / group_size)` | float32 | $\text{bias}_{i,g}$ |

So `model.embed_tokens.weight` becomes `model.embed_tokens.weight` (bytes),
`model.embed_tokens.scales`, and `model.embed_tokens.biases`. The checkpoint has no
other keys: its **stored bytes**, the sum of the sizes of its tensors, are the codes,
4 bytes per scale and per bias, and 4 bytes per gain.

The number of bits is not stored either. It follows from the shapes:
$\text{bits} = 8 \cdot \text{width of } \texttt{P.weight} \,/\, (\text{width of }
\texttt{P.scales} \cdot \text{group\_size})$.

### Example 1

```python
params = {
    "proj.weight": [
        [0.0, 0.33, -0.5, 1.0, 2.0, 2.2, 2.9, 2.6],
        [-1.0, -1.0, -1.0, -1.0, 0.5, -0.25, 0.1, 0.3],
    ],
    "norm.weight": [1.0, 0.5, 2.0, 1.5, 1.0, 1.0, 0.25, 4.0],
}
ckpt = quantize_checkpoint(params, bits=4, group_size=4)

ckpt["proj.weight"]  # [[133, 240,  48, 175],      uint8, shape (2, 4)
#  [  0,   0,  15, 183]]
ckpt["proj.scales"]  # [[0.1, 0.06], [1.0, 0.05]]
ckpt["proj.biases"]  # [[-0.5, 2.0], [-1.0, -0.25]]
ckpt["norm.weight"]  # [1.0, 0.5, 2.0, 1.5, 1.0, 1.0, 0.25, 4.0], unchanged
```

The codes are those of the example in `groupwise-4bit`:
`[[5, 8, 0, 15, 0, 3, 15, 10], [0, 0, 0, 0, 15, 0, 7, 11]]`. The first byte packs
codes 5 and 8 as $5 + 16 \cdot 8 = 133$; the last packs 7 and 11 as
$7 + 16 \cdot 11 = 183$. The first group of row 1 is constant: scale 1, codes 0, and
the bias alone reproduces $-1$. The checkpoint stores $8 + 16 + 16 + 32 = 72$ bytes;
the float parameters took 96.

## Stage 2: the forward pass on codes

`forward_q(ckpt, cfg, tokens, group_size)` returns the logits of the model whose
weights are $\hat W$: float32, shape `(B, T, vocab_size)` for integer `tokens` of
shape `(B, T)`. `ckpt` is a checkpoint in the layout of stage 1; its matrices may
have **different** numbers of bits (derive each from its shapes).

The pass is that of `decoder-forward`. Write $D$ = `d_model`, $H$ = `n_heads`,
$H_{kv}$ = `n_kv_heads`, $d$ = `head_dim`, and $\operatorname{lin}_P(x) = x\,\hat
W_P^{\top}$ for the matrix named `P.weight`:

1. **Embed.** $x_{b,t} = \hat E[\text{tokens}_{b,t}]$, a row of the dequantized
   embedding matrix; no scaling.
2. **Blocks**, for layer $i = 0, \dots,$ `n_layers` $- 1$ with the names
   `model.layers.{i}.*`:
   $x \leftarrow x + \operatorname{Attn}(\operatorname{RMSNorm}(x; g_1))$, then
   $x \leftarrow x + \operatorname{MLP}(\operatorname{RMSNorm}(x; g_2))$, with $g_1$
   = `input_layernorm.weight`, $g_2$ = `post_attention_layernorm.weight`, and
   `ops.rms_norm(x, g, cfg.norm_eps)`.
   - $\operatorname{Attn}(u)$: $q = \operatorname{lin}_{\text{q\_proj}}(u)$ split
     into $H$ heads, $k = \operatorname{lin}_{\text{k\_proj}}(u)$ and
     $v = \operatorname{lin}_{\text{v\_proj}}(u)$ split into $H_{kv}$ heads (reshape
     the last axis to `(heads, d)` and move the head axis forward: `(B, heads, T,
     d)`). Rotate $q$ and $k$, not $v$, with `ops.rope(x, positions,
     cfg.rope_base)`, positions $0, \dots, T - 1$. Attend causally with
     `ops.attention(q, k, v, mask)`, where the boolean mask lets query position $t$
     see key positions $s \le t$ (grouped-query attention is built in). Merge the
     heads back to `(B, T, D)` and apply $\operatorname{lin}_{\text{o\_proj}}$.
   - $\operatorname{MLP}(u) = \operatorname{lin}_{\text{down\_proj}}\big(
     \operatorname{silu}(\operatorname{lin}_{\text{gate\_proj}}(u)) \cdot
     \operatorname{lin}_{\text{up\_proj}}(u)\big)$ with
     $\operatorname{silu}(z) = z \cdot \sigma(z)$.
3. **Final norm** with `model.norm.weight`, then the **head**:
   $\text{logits} = \operatorname{lin}_{\text{head}}(x)$, where the head is
   `model.embed_tokens.weight` if `cfg.tie_embeddings`, else `lm_head.weight`.

**Multiplying by codes.** Compute every $\operatorname{lin}_P$ as in
`quantized-matmul`, without building $\hat W_P$: with the inputs and the codes of a
row cut into the same groups,

$$
(x\,\hat W^{\top})_i = \sum_{g} \Big( \text{scale}_{i,g} \sum_{j \in g} x_j\,\text{code}_{i,j} \;+\; \text{bias}_{i,g} \sum_{j \in g} x_j \Big).
$$

The embedding is a lookup, not a product: dequantize only the rows that are looked
up, each group of a row with its own scale and bias. Unpacking bytes to codes and
converting codes to float32 is allowed.

> [!NOTE]
> The judge compares logits, and a pass that rebuilt every float matrix first would
> produce the same numbers, so it cannot see which one you wrote. Multiplying from
> the codes is the rule of the exercise all the same: it is what makes a quantized
> model small in memory and not only on disk.

### Example 2

The kit decoder used by most cases is `cfg = DecoderConfig(d_model=64, d_mlp=128,
n_layers=2)` (64 tokens, 4 query heads, 2 key/value heads) with
`params = decoder_params(cfg, seed=1)`: 78,144 parameters. For
`tokens = [[0, 17, 42, 5, 9, 33, 17, 60]]` the first five logits at the last
position are

| weights | logits `[0, 7, :5]` |
|---|---|
| float | `-0.2050, 1.3158, -0.4835, 1.6914, 3.7705` |
| 8 bits, groups of 64 | `-0.2105, 1.2722, -0.5426, 1.6356, 3.7985` |
| 4 bits, groups of 64 | `-0.5923, 0.5637, -1.2291, 1.0923, 3.3383` |

`forward_q` must return the second or third row for the corresponding checkpoint:
the test compares with the float pass run on $\hat W$, not on $W$.

## Stage 3: cached generation

**`step_q(ckpt, cfg, tokens, cache, group_size)`** runs `T` new tokens (integer,
shape `(B, T)`, $T \ge 1$) that continue the `S` tokens a cache covers, and returns
`(logits, new_cache)`.

- `cache` is `None` ($S = 0$) or a list with one pair `(k, v)` per layer. `k` and
  `v` are float32 tensors of shape `(B, n_kv_heads, S, head_dim)`: the keys **after
  their RoPE rotation** and the values of the first `S` tokens in that layer, oldest
  first.
- The new token at index $i$ of the call has absolute position $S + i$. That is its
  RoPE position (for its query and its key), and it attends to key positions
  $j \le S + i$: every cached entry, the new entries before it, and its own.
- In each layer the new rotated keys and the new values are appended to the cached
  ones along the sequence axis (`kv-cache-append`), and the queries of the new
  tokens attend over all $S + T$ entries. Everything else is the pass of stage 2,
  applied to the new tokens only.
- `logits` is float32 of shape `(B, T, vocab_size)`. `new_cache` has the same
  structure with $S + T$ entries. The cache that was passed in is not modified: a
  caller may continue one cache twice.

Feeding a sequence in any chunks gives the logits and the final cache of one call,
and `step_q(ckpt, cfg, tokens, None, group_size)[0]` equals `forward_q(ckpt, cfg,
tokens, group_size)`.

**`generate_q(ckpt, cfg, prompt, max_new_tokens, group_size, cache=None)`** decodes
greedily. `prompt` is a list of $P \ge 1$ ints. If `cache` is given, it is a cache
for one sequence ($B = 1$) and the prompt continues the tokens it covers; those
tokens themselves are not available. Each new token is the argmax of the logits
after everything before it (ties to the lowest id); it is appended and fed back.
Returns the `max_new_tokens` $\ge 1$ generated tokens as a list of Python ints.

**Cost.** Every token goes through the model once: the prompt in one call, then one
token per step, and nothing after the last generated token. The tests count the
query positions handed to `ops.attention` (batch size times `T`, summed over calls)
while `generate_q` runs: at most `n_layers * (P + max_new_tokens - 1)`.

### Example 3

With the model of Example 2 quantized to 4 bits in groups of 64,
`generate_q(ckpt, cfg, [0, 5, 24, 41, 8], 4, 64)` returns `[51, 2, 52, 53]` and asks
attention for $2 \cdot (5 + 3) = 16$ query positions. Recomputing the whole sequence
for every token would ask for $2 \cdot (5 + 6 + 7 + 8) = 52$. (The float model
continues the same prompt with `[51, 43, 57, 9]`: at 4 bits the second token is
already another one.)

## Stage 4: a mixed recipe under a budget

**`bits_per_weight(ckpt, group_size)`** returns, as a Python float,

$$
\frac{\text{stored bits}}{\text{parameters}} .
$$

Stored bits are 8 times the stored bytes of stage 1: 8 per byte of packed codes and
32 per scale, bias, and gain. Parameters are the elements of the float tensors the
checkpoint stands for: `out * in` for every quantized matrix (`in` is the width of
its `scales` times `group_size`) plus the elements of every tensor that was stored
unchanged. Scales and biases are overhead, not parameters.

**`quantize_mixed(params, group_size, max_bpw)`** returns the checkpoint of this
recipe, in the layout of stage 1:

- `model.embed_tokens.weight` and, if present, `lm_head.weight`: **8 bits**;
- every other matrix: **4 bits**;
- gains unchanged.

If `bits_per_weight` of that checkpoint is greater than `max_bpw`, the function
raises `ValueError` instead of returning it. A checkpoint of exactly `max_bpw` fits.

### Example 4

The model of Example 2, groups of 64:

| checkpoint | stored bytes | bits per weight | mean KL from the float model |
|---|---|---|---|
| float32 | 312,576 | 32 | 0 |
| 8 bits everywhere | 88,832 | 9.094 | 0.0006 |
| the recipe | 51,968 | 5.320 | 0.105 |
| 4 bits everywhere | 49,920 | 5.111 | 0.176 |

The KL column is for `decoder_params(cfg, seed=3)`, the model of the public case.
"4 bits" costs 5.11 bits per weight: 4 for the code, 1 for the two float32 numbers
that every group of 64 shares, and the rest for the gains. The embedding is 5 percent
of the weights, so giving it 8 bits adds 0.21 bits per weight, and it removes 40
percent of the divergence, because every token enters through one of its rows and
(tied) leaves through all of them. `quantize_mixed(params, 64, 5.5)` returns the
recipe; with `max_bpw = 5.25` it raises `ValueError`.

## Constraints

- `bits` is 4 or 8. `group_size` divides the `in` of every matrix and `in * bits` is
  a multiple of 8; the tests use 32 and 64 (and 4 in Example 1). Weights are finite,
  $|w| \le 100$; groups may be constant. No output may be NaN or infinite.
- **Stage 1.** Codes are compared exactly, with one allowance: where the exact
  quotient $(w - \text{bias}) / \text{scale}$ lies within $10^{-3}$ of a tie
  ($k + 0.5$), either neighboring code is accepted, since a float32 division that is
  one bit off may round the other way. Packed codes must be uint8 with the specified
  shape. Scales and biases: relative tolerance `1e-4`, absolute `1e-6`. Gains:
  exactly. The stored bytes must not exceed the format's: 49,920 bytes for the model
  of Example 2 at 4 bits and 88,832 at 8. Property: every $\hat w$ is within half a
  step ($\text{scale} / 2$) of $w$, and the minimum and maximum of every group come
  back exactly.
- **Stages 2 and 3.** Logits and cached keys and values are compared with the float
  pass on $\hat W$ to a relative tolerance of `1e-3` and an absolute tolerance of
  `2e-4`. $1 \le B \le 2$, $T \le 8$, $S + T \le 16$. Generated tokens are compared
  exactly; wherever an argmax is taken, the two largest logits differ by at least
  `1e-2`.
- **Stage 3, cost.** At most `n_layers * (P + max_new_tokens - 1)` query positions
  through `ops.attention`, as specified above.
- **Stage 4.** `bits_per_weight` is compared to a relative tolerance of `1e-9`. The
  recipe's checkpoint is compared as in stage 1. On the public model (seed 3, groups
  of 64, `max_bpw = 5.5`) and on the hidden ones, the true size of your checkpoint is
  at most `max_bpw`, and the KL divergence of `forward_q` on your checkpoint from
  the float model's logits, averaged over 64 positions, is at most **0.14 nats**.
  The recipe measures 0.105 to 0.115 nats on these models, and 4 bits everywhere
  0.175 to 0.180.
- The time limit per case is 30 seconds; the reference needs a few milliseconds per
  case in PyTorch and MLX. In JAX, every new combination of shapes is compiled once,
  which takes a few seconds in total on a first run.

## Why this matters

This is the path a model takes into `mlx-lm` or llama.cpp: a converter writes
packed codes with per-group scales and biases, and a runtime that never sees the
float weights again embeds, multiplies, caches, and samples from them. Each stage
has a failure that only shows at the next one. A checkpoint that keeps a float
matrix "for the lookup" is the size of a model that was never quantized. A matmul
that assumes one bit width, or one group per row, runs the uniform checkpoint and
breaks on the mixed one that everybody actually ships. A cache with positions that
restart, or keys stored in another convention than the step expects, produces fluent
text that differs from the uncached pass. And a recipe judged by its nominal bits
is over budget by exactly the metadata nobody counted: "4-bit" is 4.5 bits per
weight in MLX and 5.1 here, and the number that has to fit in memory is the real
one.
