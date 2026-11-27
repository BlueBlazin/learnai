Build a runner for a quantized transformer decoder in four stages. You will convert a float32 checkpoint into packed 4-bit or 8-bit codes with per-group scales and biases, run the forward pass directly from those packed codes, generate tokens through a KV cache, and build a mixed-precision recipe that meets a bits-per-weight budget without degrading predictions.

## Overview

```python
def quantize_checkpoint(params, bits, group_size): ...  # stage 1 -> checkpoint
def forward_q(ckpt, cfg, tokens, group_size): ...  # stage 2 -> logits
def step_q(ckpt, cfg, tokens, cache, group_size): ...  # stage 3 -> (logits, cache)
def generate_q(ckpt, cfg, prompt, max_new_tokens, group_size, cache=None): ...  # stage 3 -> tokens
def bits_per_weight(ckpt, group_size): ...  # stage 4 -> float
def quantize_mixed(params, group_size, max_bpw): ...  # stage 4 -> checkpoint
```

![Left, the checkpoint that stages 1 and 4 write. Right, one step of stages 2 and 3, which reads only that checkpoint.](assets/runner.svg)

Test cases are grouped by stage, and each stage uses only what that stage and earlier ones define. In stages 2 and 3, tests pass checkpoints built by the judge under the stage 1 format, so your forward pass cannot rely on any extra keys or float weights. Stage 4 runs your own mixed-precision checkpoint through your own forward pass.

### Model architecture and weights

The model is the kit decoder from `decoder-forward`, configured by a [`DecoderConfig`](../../../python/src/learnai/kit/decoder.py) `cfg` with fields `vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_mlp`, `rope_base`, `norm_eps`, `tie_embeddings`, and the derived `head_dim = d_model // n_heads`. `params` maps parameter names to float32 tensors. All linear weights are `(out, in)` matrices without biases:

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
| `lm_head.weight` (only when `cfg.tie_embeddings` is `False`) | `(vocab_size, d_model)` |

You may use `ops.rms_norm`, `ops.rope`, and `ops.attention` from the kit's [`ops`](../../../python/src/learnai/kit/ops/__init__.py). Do not call the kit's reference forward passes (`decoder_forward_ref` or `decoder_block_ref`).

## Stage 1: the quantized checkpoint

`quantize_checkpoint(params, bits, group_size)` returns a new dictionary representing the **checkpoint**, where `bits` is `4` or `8`. Every 2D tensor in `params` is treated as a weight matrix and quantized. Every other tensor (such as 1D norm gains) is stored unchanged under its original key.

### Groupwise quantization

Quantize each weight matrix $w$ of shape `(out, in)` as in `groupwise-4bit`. Split each row along the input axis into `in / group_size` consecutive groups, so group $g$ of row $i$ contains columns $j$ with $g \cdot \text{group\_size} \le j < (g + 1) \cdot \text{group\_size}$. Let $L = 2^{\text{bits}} - 1$ be the maximum integer code. For each group, compute the bias (minimum value) and scale (step size):

$$
\text{bias}_{i,g} = \min_{j \in g} w_{i,j},
\qquad
\text{scale}_{i,g} = \begin{cases}
(\max_{j \in g} w_{i,j} - \min_{j \in g} w_{i,j}) / L & \text{if } \max_{j \in g} w_{i,j} > \min_{j \in g} w_{i,j}, \\
1 & \text{if every element in the group is equal}.
\end{cases}
$$

Then quantize each weight to an integer code in `[0, L]`, which represents the dequantized value $\hat w_{i,j}$:

$$
\text{code}_{i,j} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{i,j} - \text{bias}_{i,g}}{\text{scale}_{i,g}}\Big),\ 0,\ L\Big),
\qquad
\hat w_{i,j} = \text{code}_{i,j} \cdot \text{scale}_{i,g} + \text{bias}_{i,g} .
$$

Here $\operatorname{round}$ rounds to the nearest integer with ties to even (the default `round` in PyTorch, JAX, and MLX). The dequantized weight $\hat w$ is never stored in the checkpoint.

### Bit packing and checkpoint layout

Pack each row's codes into `uint8` bytes:

- When `bits == 8`, store one code per byte, giving `in` bytes per row.
- When `bits == 4`, pack two codes per byte with the even-indexed code in the low nibble, as in `bit-packing`. Byte $b$ of a row equals $\text{code}_{2b} + 16 \cdot \text{code}_{2b+1}$, giving `in / 2` bytes per row.

For a weight matrix named `P.weight`, store exactly three entries in the checkpoint and omit the original float matrix:

| key | shape | dtype | content |
|---|---|---|---|
| `P.weight` | `(out, in * bits / 8)` | uint8 | packed codes |
| `P.scales` | `(out, in / group_size)` | float32 | $\text{scale}_{i,g}$ |
| `P.biases` | `(out, in / group_size)` | float32 | $\text{bias}_{i,g}$ |

For example, `model.embed_tokens.weight` produces `model.embed_tokens.weight` (uint8), `model.embed_tokens.scales`, and `model.embed_tokens.biases`. Store no other keys. A checkpoint's **stored bytes** is the sum of tensor byte sizes: 1 byte per packed uint8 element, 4 bytes per scale and bias, and 4 bytes per float32 gain.

The bit width is not stored as a separate field. Derive it from the tensor shapes (where `width` is the last dimension, or column count):

$$
\text{bits} = \frac{8 \cdot \text{width}(\texttt{P.weight})}{\text{width}(\texttt{P.scales}) \cdot \text{group\_size}} .
$$

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

The unpacked codes match the example in `groupwise-4bit`: `[[5, 8, 0, 15, 0, 3, 15, 10], [0, 0, 0, 0, 15, 0, 7, 11]]`. The first byte packs `5` and `8` as $5 + 16 \cdot 8 = 133$, and the last byte packs `7` and `11` as $7 + 16 \cdot 11 = 183$. Because the first group of row 1 is constant, its scale is `1.0`, its codes are `0`, and its bias `-1.0` reproduces every element exactly. The checkpoint occupies $8 + 16 + 16 + 32 = 72$ bytes compared to 96 bytes for the float parameters.

## Stage 2: the forward pass on codes

`forward_q(ckpt, cfg, tokens, group_size)` runs the decoder on integer `tokens` of shape `(B, T)` (batch size $B$, sequence length $T$) using the dequantized weights $\hat W$ represented by `ckpt`. It returns float32 logits of shape `(B, T, vocab_size)`. Different matrices in `ckpt` may use **different** bit widths, so derive `bits` separately for each matrix from its shapes.

Let $D = \texttt{d\_model}$, $H = \texttt{n\_heads}$, $H_{kv} = \texttt{n\_kv\_heads}$, and $d = \texttt{head\_dim}$. For any matrix `P.weight`, write $\operatorname{lin}_P(x) = x\,\hat W_P^{\top}$. The forward pass matches `decoder-forward`:

1. **Embedding.** Look up $x_{b,t} = \hat E[\text{tokens}_{b,t}]$ from the dequantized rows of `model.embed_tokens.weight` with no scaling.
2. **Decoder blocks.** For layer $i = 0, \dots, \texttt{n\_layers} - 1$ using weights under `model.layers.{i}.*`, update $x \leftarrow x + \operatorname{Attn}(\operatorname{RMSNorm}(x; g_1))$ and then $x \leftarrow x + \operatorname{MLP}(\operatorname{RMSNorm}(x; g_2))$. Here $g_1$ is `input_layernorm.weight`, $g_2$ is `post_attention_layernorm.weight`, and normalization uses `ops.rms_norm(x, g, cfg.norm_eps)`.
   - **Attention** $\operatorname{Attn}(u)$: Project $q = \operatorname{lin}_{\text{q\_proj}}(u)$ into $H$ heads, and $k = \operatorname{lin}_{\text{k\_proj}}(u)$ and $v = \operatorname{lin}_{\text{v\_proj}}(u)$ into $H_{kv}$ heads. Reshape the last axis to `(heads, d)` and transpose to `(B, heads, T, d)`. Apply `ops.rope(x, positions, cfg.rope_base)` with positions `0, ..., T - 1` to $q$ and $k$ (not $v$). Compute causal attention via `ops.attention(q, k, v, mask)` with a boolean mask allowing query position $t$ to attend to key positions $s \le t$ (`ops.attention` handles grouped-query attention internally). Merge heads back to `(B, T, D)` and apply $\operatorname{lin}_{\text{o\_proj}}$.
   - **MLP** $\operatorname{MLP}(u)$: Compute $\operatorname{lin}_{\text{down\_proj}}\big(\operatorname{silu}(\operatorname{lin}_{\text{gate\_proj}}(u)) \cdot \operatorname{lin}_{\text{up\_proj}}(u)\big)$, where $\operatorname{silu}(z) = z \cdot \sigma(z)$.
3. **Final norm and LM head.** Apply `ops.rms_norm` with `model.norm.weight`, then project to logits with $\operatorname{lin}_{\text{head}}(x)$. Use `model.embed_tokens.weight` as the head when `cfg.tie_embeddings` is `True`, and `lm_head.weight` otherwise.

### Multiplying directly from codes

Compute every linear projection $\operatorname{lin}_P$ as in `quantized-matmul` without materializing the full float weight matrix $\hat W_P$. Grouping the inputs $x$ and unpacked codes along the input axis gives:

$$
(x\,\hat W^{\top})_i = \sum_{g} \Big( \text{scale}_{i,g} \sum_{j \in g} x_j\,\text{code}_{i,j} \;+\; \text{bias}_{i,g} \sum_{j \in g} x_j \Big).
$$

For the embedding lookup, index into the packed bytes, scales, and biases first, then dequantize only the selected rows using each group's scale and bias. Unpacking bytes into float32 codes is allowed.

> [!NOTE]
> Because the judge checks output logits, materializing $\hat W_P$ first would produce identical numbers. Computing products directly from codes without building $\hat W_P$ is still a requirement of the problem, because that is what keeps quantized models small in memory during inference.

### Example 2

Most tests use `cfg = DecoderConfig(d_model=64, d_mlp=128, n_layers=2)` (vocab size 64, 4 query heads, 2 KV heads) with `params = decoder_params(cfg, seed=1)` (78,144 parameters). For `tokens = [[0, 17, 42, 5, 9, 33, 17, 60]]`, the first five logits at the last position `[0, 7, :5]` are:

| weights | logits `[0, 7, :5]` |
|---|---|
| float | `-0.2050, 1.3158, -0.4835, 1.6914, 3.7705` |
| 8 bits, groups of 64 | `-0.2105, 1.2722, -0.5426, 1.6356, 3.7985` |
| 4 bits, groups of 64 | `-0.5923, 0.5637, -1.2291, 1.0923, 3.3383` |

`forward_q` must match the second or third row for the corresponding quantized checkpoint, because the tests compare against the float forward pass run on $\hat W$, not on $W$.

## Stage 3: cached generation

### Single step with KV cache

`step_q(ckpt, cfg, tokens, cache, group_size)` processes $T \ge 1$ new tokens (integer tensor of shape `(B, T)`) that follow $S$ previously cached tokens, and returns `(logits, new_cache)`.

- `cache` is either `None` ($S = 0$) or a list containing one `(k, v)` pair per layer. Each `k` and `v` is a float32 tensor of shape `(B, n_kv_heads, S, head_dim)` storing the keys **after RoPE rotation** and the values of the first $S$ tokens, oldest first.
- New token index $i \in \{0, \dots, T - 1\}$ has absolute position $S + i$. Use those absolute positions for RoPE on the new queries and keys, and mask attention so query $i$ sees all key positions $j \le S + i$ (all cached keys, earlier new keys, and itself).
- In each layer, append the newly rotated keys and new values to the cached tensors along the sequence axis (`kv-cache-append`), then attend over all $S + T$ entries. All other operations match stage 2 and run only on the $T$ new tokens.
- `logits` is float32 with shape `(B, T, vocab_size)`, and `new_cache` has the same structure as `cache` with sequence length $S + T$. Do not mutate the input `cache`, because callers may branch from the same cache more than once.

Processing a sequence in arbitrary chunks must produce the same logits and final cache as processing it in one call, and `step_q(ckpt, cfg, tokens, None, group_size)[0]` must equal `forward_q(ckpt, cfg, tokens, group_size)`.

### Greedy generation

`generate_q(ckpt, cfg, prompt, max_new_tokens, group_size, cache=None)` runs greedy decoding. `prompt` is a list of $P \ge 1$ Python ints. When `cache` is provided, it holds a single sequence ($B = 1$) whose cached prefix precedes `prompt`, and the prefix tokens themselves are not passed in. At each step, pick the argmax of the logits at the last position (breaking ties toward the lowest token id), append that token to the output, and feed it into the next step. Return the `max_new_tokens` $\ge 1$ generated tokens as a list of Python ints.

Every token must pass through the model exactly once: prefill the entire prompt in one `step_q` call, then decode one token per step without calling `step_q` after the final generated token. Tests count total query positions passed to `ops.attention` (summing `B * T` across calls), which must be at most `n_layers * (P + max_new_tokens - 1)`.

### Example 3

Using the Example 2 model quantized to 4 bits with `group_size = 64`, calling `generate_q(ckpt, cfg, [0, 5, 24, 41, 8], 4, 64)` returns `[51, 2, 52, 53]` and passes $2 \cdot (5 + 3) = 16$ query positions to `ops.attention`. Recomputing the full sequence at every step would pass $2 \cdot (5 + 6 + 7 + 8) = 52$ query positions. (For comparison, the unquantized float model generates `[51, 43, 57, 9]`, diverging on the second token.)

## Stage 4: a mixed recipe under a budget

### Measuring bits per weight

`bits_per_weight(ckpt, group_size)` returns a Python float equal to:

$$
\frac{\text{stored bits}}{\text{parameters}} .
$$

Stored bits equal 8 times the stored bytes defined in stage 1: 8 bits per uint8 element of packed codes, and 32 bits per float32 scale, bias, or gain. Parameters equal the total number of float weights the checkpoint represents: `out * in` for each quantized matrix (where `in = scales.shape[1] * group_size`) plus the element count of every unquantized tensor. Scales and biases count as storage overhead, not as parameters.

### Mixed-precision quantization

`quantize_mixed(params, group_size, max_bpw)` builds a checkpoint in the stage 1 format using this recipe:

- Quantize `model.embed_tokens.weight` and, if present, `lm_head.weight` to **8 bits**.
- Quantize every other 2D weight matrix to **4 bits**.
- Keep 1D norm gains unchanged.

If `bits_per_weight(ckpt, group_size) > max_bpw`, raise `ValueError`. A checkpoint whose bits per weight equals `max_bpw` fits and is returned.

### Example 4

For the Example 2 architecture with `group_size = 64` (and KL divergence measured on `decoder_params(cfg, seed=3)`):

| checkpoint | stored bytes | bits per weight | mean KL from the float model |
|---|---|---|---|
| float32 | 312,576 | 32 | 0 |
| 8 bits everywhere | 88,832 | 9.094 | 0.0006 |
| the recipe | 51,968 | 5.320 | 0.105 |
| 4 bits everywhere | 49,920 | 5.111 | 0.176 |

Uniform 4-bit quantization actually costs 5.111 bits per weight: 4 bits for each code, 1 bit per weight for two float32 metadata values shared across 64 weights, plus the unquantized gains. Because the embedding is 5% of all parameters, raising it to 8 bits adds only 0.209 bits per weight while cutting KL divergence by 40% (every token enters through an embedding row and, when tied, exits through all rows). Thus `quantize_mixed(params, 64, 5.5)` succeeds, while `max_bpw = 5.25` raises `ValueError`.

## Constraints

- `bits` is `4` or `8`. `group_size` divides the `in` dimension of every matrix, and `in * bits` is divisible by 8. Tests use `group_size` 32 and 64 (and 4 in Example 1). All weights are finite with $|w| \le 100$, and groups may be constant. No output may contain NaN or Inf.
- **Stage 1.** Unpacked codes are compared exactly, except where the exact quotient $(w - \text{bias}) / \text{scale}$ lies within $10^{-3}$ of a half-integer tie ($k + 0.5$), where either neighboring integer is accepted. Packed codes must be `uint8` with the exact required shape. Scales and biases are checked with `rtol = 1e-4` and `atol = 1e-6`, and gains are checked exactly. Stored bytes must not exceed the format size (49,920 bytes for Example 2 at 4 bits and 88,832 bytes at 8 bits). Every $\hat w$ must lie within $\text{scale} / 2$ of $w$, and each group's minimum and maximum must reconstruct exactly.
- **Stages 2 and 3.** Logits and cached keys and values are compared against the float forward pass on $\hat W$ with `rtol = 1e-3` and `atol = 2e-4`. Tests use $1 \le B \le 2$, $T \le 8$, and $S + T \le 16$. Generated tokens are compared exactly, and the top two logits at every argmax differ by at least `1e-2`.
- **Stage 3 cost.** At most `n_layers * (P + max_new_tokens - 1)` query positions may be passed to `ops.attention`.
- **Stage 4.** `bits_per_weight` is checked with `rtol = 1e-9`. The mixed checkpoint is checked with the stage 1 rules. On the public model (`seed = 3`, `group_size = 64`, `max_bpw = 5.5`) and hidden models, your checkpoint's true bits per weight must be at most `max_bpw`, and the mean KL divergence of `forward_q` from the float model over 64 positions must be at most **0.14 nats** (the recipe achieves 0.105 to 0.115 nats, compared to 0.175 to 0.180 nats for uniform 4-bit).
- The time limit per case is 30 seconds. The reference takes a few milliseconds per case in PyTorch and MLX, and a few seconds total on the first run in JAX due to compilation.

## Why this matters

This pipeline mirrors how `mlx-lm` and `llama.cpp` deploy quantized models. A converter writes packed codes plus per-group scales and biases, and a runtime embeds, multiplies, caches, and decodes directly from that format without ever seeing float weights again.

Silent bugs in one stage often surface only in the next. Keeping a float copy of the embedding table bloats checkpoint size, and hardcoding a single bit width passes uniform tests but breaks on real mixed-precision checkpoints. Restarting RoPE positions or storing unrotated keys corrupts cached generation with fluent-looking wrong text. Omitting scale and bias overhead underestimates true bits per weight (4-bit quantization is 4.5 bits per weight in MLX with float16 metadata and 5.1 here with float32 metadata).
