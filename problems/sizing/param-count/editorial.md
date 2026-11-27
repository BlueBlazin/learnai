## The idea

Every parameter in a pre-norm decoder belongs to a specific tensor. Each transformer block owns four attention projection matrices, two or three MLP matrices, and two norm vectors. Outside the blocks, the model adds one token embedding table, one final norm, and optionally a separate output head. A linear layer mapping `n_in` features to `n_out` features contributes `n_in * n_out` weights plus `n_out` biases when bias is enabled.

**Attention.** With residual width $d$, $H$ query heads of width $h$, and $H_{kv}$ key/value heads of width $h$, the query projection outputs $H h$ features while the key and value projections each output $H_{kv} h$ features. Grouped-query attention (Ainslie et al., 2023) lets $H / H_{kv}$ query heads share one key head and one value head, shrinking $W_k$ and $W_v$ by that factor while $W_q$ and $W_o$ stay full size:

$$
\text{attention} = \underbrace{d \cdot H h}_{W_q} + \underbrace{2\, d \cdot H_{kv} h}_{W_k,\,W_v}
+ \underbrace{H h \cdot d}_{W_o}.
$$

Do not substitute $d$ for $H h$. They are equal in many models, but not all. In Qwen3-0.6B, $H h = 16 \cdot 128 = 2048$ while $d = 1024$. In Gemma 3 4B, $H h = 8 \cdot 256 = 2048$ while $d = 2560$.

**MLP.** A standard MLP computes `down(act(up(x)))` using two matrices. A gated MLP (such as SwiGLU in Llama, Qwen, and Granite, or GeGLU in Gemma) computes `down(act(gate(x)) * up(x))` using three matrices. To match the parameter count of a standard MLP with hidden width $f = 4d$, gated models often pick $f \approx \tfrac{8}{3} d$, though architectures are free to vary. Granite 4.2 3B, for example, uses $f = 3.2\,d$. Some checkpoints store the gate and up weights fused in a single `(2f, d)` tensor (for example, Granite 4.0's `shared_mlp.input_linear` has shape `[4096, 1024]`), which has the same total parameter count.

**Tied embeddings.** For vocabulary size $V$, the output head computes logits with a `(V, d)` matrix, which has the same shape as the token embedding table. When embeddings are tied, both layers point to the same tensor in memory, so it is counted once. With vocabularies of 100k to 260k tokens, this single table is often the largest tensor in a small model, accounting for 29 percent of Granite 4.0 350M.

## Reference solution (Python)

```python
def count_params(cfg):
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]

    attention = d * q_width + 2 * d * kv_width + q_width * d
    if cfg["attn_bias"]:
        attention += q_width + 2 * kv_width + d

    d_ff = cfg["d_ff"]
    if cfg["mlp_type"] == "swiglu":
        mlp, mlp_bias = 3 * d * d_ff, 2 * d_ff + d
    else:
        mlp, mlp_bias = 2 * d * d_ff, d_ff + d
    if cfg["mlp_bias"]:
        mlp += mlp_bias

    per_norm = 2 * d if cfg["norm_bias"] else d
    n_layers = cfg["n_layers"]
    counts = {
        "embedding": cfg["vocab_size"] * d,
        "attention": n_layers * attention,
        "mlp": n_layers * mlp,
        "norm": (2 * n_layers + 1) * per_norm,
        "head": 0 if cfg["tie_embeddings"] else cfg["vocab_size"] * d,
    }
    counts["total"] = sum(counts.values())
    return counts
```

## Checking against real checkpoints

The example configurations come from each repository's `config.json` on Hugging Face (`https://huggingface.co/<repo>/raw/main/config.json`), using `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, and `tie_word_embeddings`. Granite omits `head_dim`, so it defaults to `hidden_size / num_attention_heads = 64`. The published counts below come from the safetensors metadata shown on each model page on 2026-11-14.

| model | this function | published | note |
|---|---|---|---|
| `ibm-granite/granite-4.0-350m` | 352,379,904 | 352,379,904 | tied |
| `ibm-granite/granite-4.2-3b` | 3,659,737,600 | 3,659,737,600 | untied |
| `Qwen/Qwen3-4B-Instruct-2507` | 4,022,458,880 | 4,022,468,096 | tied; differs by QK-norm |
| `Qwen/Qwen3-0.6B` | 596,042,752 | 751,632,384 | tied; differs by QK-norm and a duplicate |

The two Qwen3 rows show why exact checks are useful. Qwen3-4B has 36 blocks of width 2560, 32 query heads, 8 KV heads of width 128, a SwiGLU width of 9728, and a vocabulary of 151,936. Qwen3-0.6B has 28 blocks of width 1024, 16 query heads, 8 KV heads of width 128, and a SwiGLU width of 3072. Both models add two RMSNorm gain vectors of length `head_dim` per block for QK-norm, which this problem omits. For Qwen3-4B, those gains total $36 \cdot 2 \cdot 128 = 9216$ parameters, which is the exact difference. For Qwen3-0.6B, the difference is $28 \cdot 2 \cdot 128 = 7168$ **plus 155,582,464** ($151936 \cdot 1024$). That checkpoint stores the tied embedding matrix under both tensor names, so the Hub counts it twice and reports 751.6 M for a 596.0 M-parameter model.

## Common mistakes

- **Counting a tied head twice.** Adding $V d$ for both `embedding` and `head` regardless of `tie_embeddings` overcounts by $V d$. For Granite 4.0 350M, that adds 103 M to a 352 M model.
- **Sizing K and V like Q.** Using $H h$ for all four attention projections ignores grouped-query attention. For Granite 4.2 3B, this overcounts attention by 67 percent (419 M parameters).
- **Counting a gated MLP as two matrices.** Using `2 * d * d_ff` for SwiGLU drops one-third of the MLP, which is the largest component of the model. For Granite 4.2 3B, that misses 839 M of 3.66 B parameters.
- **Assuming `n_heads * head_dim == d_model`.** Writing `4 * d * d` for attention or deriving KV width as `d * n_kv_heads // n_heads` works for Granite, but fails for Qwen3-0.6B and Gemma 3. Always use `head_dim` directly.
- **Forgetting the final norm.** For $L$ blocks, there are $2L + 1$ norms in total, not $2L$. Missing the final norm leaves the total short by $d$ parameters, which immediately shows up when comparing against a checkpoint.
- **Dropping biases or using the wrong bias length.** Every bias vector has one entry per *output* feature: $H h$ for $W_q$, $H_{kv} h$ for $W_k$ and $W_v$, $d$ for $W_o$, and $f$ for each MLP up/gate projection. Because modern decoders rarely use biases, code that ignores the bias flags looks right until it encounters an older or vision-language model.

## Complexity

$O(1)$ time and space, using a fixed number of arithmetic operations. Python integers do not overflow, though in fixed-width languages you need 64-bit integers because parameter counts pass $2^{32}$ around 4.3 B parameters.

## In the wild

Multiplying the parameter count by the bytes per weight gives the resident model size. Granite 4.2 3B takes $3.66 \cdot 10^9 \cdot 2 = 7.32$ GB in BF16, matching the sum of its two safetensors shards. An 8-bit quantized build takes about 3.89 GB instead of 3.66 GB because each group of 64 weights also stores a scale and a bias.

The component breakdown matters as much as the total. The MLP holds 69 percent of Granite 4.2 3B's parameters, making it the main target for quantization, whereas the embedding table is read one row per token and costs memory capacity rather than bandwidth. Tools that report parameter counts (`model.num_parameters()`, the Hugging Face safetensors widget, `llama.cpp`'s load log) handle tied and duplicated tensors differently, so when two tools disagree, the gap is usually $V d$.
