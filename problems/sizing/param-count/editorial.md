## The idea

Count tensors, not layers. A decoder block owns four attention matrices, two or three
MLP matrices, and two norm vectors; the model adds one embedding table, one final
norm, and possibly a separate head. Each matrix from `n_in` to `n_out` features has
`n_in * n_out` weights and, optionally, `n_out` biases. Everything else is deciding
what `n_in` and `n_out` are.

**Attention.** With $H$ query heads of width $h$ and $H_{kv}$ key/value heads, the
query projection produces $H h$ features and the key and value projections produce
$H_{kv} h$ each. Grouped-query attention (Ainslie et al., 2023) lets $H / H_{kv}$
query heads share one key head and one value head, so $W_k$ and $W_v$ shrink by that
factor while $W_q$ and $W_o$ stay full size:

$$
\text{attention} = \underbrace{d \cdot H h}_{W_q} + \underbrace{2\, d \cdot H_{kv} h}_{W_k,\,W_v}
+ \underbrace{H h \cdot d}_{W_o}.
$$

Do not substitute $d$ for $H h$. In most models they are equal, but in Qwen3-0.6B
$H h = 16 \cdot 128 = 2048$ while $d = 1024$, and in Gemma 3 4B $H h = 8 \cdot 256 =
2048$ while $d = 2560$.

**MLP.** A classic MLP is `down(act(up(x)))`: two matrices. A gated MLP (SwiGLU in
Llama, Qwen, and Granite; GeGLU in Gemma) is `down(act(gate(x)) * up(x))`: three. To
keep the parameter count of a classic MLP with $f = 4d$, gated models usually choose
$f \approx \tfrac{8}{3} d$, but nothing forces that: Granite 4.2 3B uses $f = 3.2\,d$.
Some checkpoints store gate and up fused in one `2f x d` tensor (Granite 4.0's
`shared_mlp.input_linear` has shape `[4096, 1024]`); the count is the same.

**Tied embeddings.** The head computes logits as $h W^\top$ with a $V \times d$ matrix,
the same shape as the embedding table. A tied model uses the same tensor for both, so
it exists once. With a 100k to 260k vocabulary this is the largest single tensor in a
small model: 29 percent of Granite 4.0 350M.

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

The configurations in the statement come from each repository's `config.json`
(`https://huggingface.co/<repo>/raw/main/config.json`: `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
`intermediate_size`, `vocab_size`, `tie_word_embeddings`; Granite has no `head_dim`
key, it is `hidden_size / num_attention_heads = 64`). The published counts come from
the safetensors metadata that the Hub shows on the model page, read on 2026-11-14.

| model | this function | published | note |
|---|---|---|---|
| `ibm-granite/granite-4.0-350m` | 352,379,904 | 352,379,904 | tied |
| `ibm-granite/granite-4.2-3b` | 3,659,737,600 | 3,659,737,600 | untied |
| `Qwen/Qwen3-4B-Instruct-2507` | 4,022,458,880 | 4,022,468,096 | tied; differs by QK-norm |
| `Qwen/Qwen3-0.6B` | 596,042,752 | 751,632,384 | tied; differs by QK-norm and a duplicate |

The Qwen3 rows are worth working through. Qwen3 (36 blocks of width 2560, 32 heads
over 8 KV heads of width 128, SwiGLU 9728, vocabulary 151,936 for the 4B model; 28
blocks of width 1024, 16 over 8 heads of width 128, SwiGLU 3072 for the 0.6B model)
normalizes queries and keys with two extra RMSNorm gains of `head_dim` per block,
which this problem leaves out: $36 \cdot 2 \cdot 128 = 9216$ for the 4B model, exactly
the gap. For the 0.6B model the gap is $28 \cdot 2 \cdot 128 = 7168$ **plus
155,582,464**, which is $151936 \cdot 1024$: that checkpoint file stores the tied
matrix under both names, so the Hub counts it twice and reports 751.6 M for a model
with 596.0 M parameters. It is the "tied head counted twice" mistake, made by a tool.

## Common mistakes

- **Counting a tied head twice.** `embedding + head` with both at $V d$, whatever
  `tie_embeddings` says. Off by $V d$: 103 M of 352 M for Granite 4.0 350M.
- **Sizing K and V like Q.** Using $H h$ for all four projections ignores
  grouped-query attention. For Granite 4.2 3B it overcounts attention by 67 percent
  (419 M parameters).
- **Counting a gated MLP as two matrices.** `2 * d * d_ff` for SwiGLU loses a third of
  the MLP, which is where most parameters live: 839 M of 3.66 B for Granite 4.2 3B.
- **Assuming `n_heads * head_dim == d_model`.** Writing `4 * d * d` for attention, or
  deriving the KV width as `d * n_kv_heads // n_heads`, is right for Granite and wrong
  for Qwen3-0.6B and Gemma 3. The config has a `head_dim` field for a reason.
- **Forgetting the final norm.** There are $2L + 1$ norms, not $2L$. It is only $d$
  parameters, which is exactly why an exact comparison with the checkpoint is useful:
  a count that is off by 2560 tells you which tensor you missed.
- **Dropping biases, or giving a bias the wrong length.** A bias has one entry per
  *output* feature: $W_q$'s has $H h$, $W_k$'s has $H_{kv} h$, the MLP's up
  projection's has $f$. Modern decoders mostly have none, so code that ignores the
  flags looks right until it meets an older or a vision-language model.

## Complexity

$O(1)$: a fixed number of multiplications. Python ints do not overflow; in a language
with fixed-width integers use 64 bits, since totals pass $2^{32}$ at about 4.3 B
parameters.

## In the wild

Multiply the total by the bytes per weight to get the resident size: Granite 4.2 3B is
$3.66 \cdot 10^9 \cdot 2 = 7.32$ GB in BF16, and its two safetensors shards add up to
7.32 GB. An 8-bit build is about 3.89 GB rather than 3.66 GB because each group of 64
weights also stores a scale and a bias. The breakdown matters as much as the total:
the MLP holds 69 percent of Granite 4.2 3B, so that is where quantization pays, while
the embedding table is read one row per token and costs memory but almost no
bandwidth. Tools that print these numbers (`model.num_parameters()`, the Hub's
safetensors widget, `llama.cpp`'s load log) each make their own choice about tied and
duplicated tensors, so when two of them disagree, the difference is usually $V d$.
