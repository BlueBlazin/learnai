Compute the parameter count of a decoder-only transformer from its configuration dictionary, broken down by component.

## Specification

```python
def count_params(cfg: dict) -> dict[str, int]: ...
```

`cfg` describes a pre-norm decoder-only transformer with rotary position embeddings (RoPE), which have no learnable parameters. It is a dictionary with these exact keys:

| key | type | meaning |
|---|---|---|
| `vocab_size` | `int` | vocabulary size $V$ |
| `d_model` | `int` | residual stream width $d$ |
| `n_layers` | `int` | number of transformer blocks $L$ |
| `n_heads` | `int` | number of query heads $H$ |
| `n_kv_heads` | `int` | number of key and value heads $H_{kv}$, which divides `n_heads`. When $H_{kv} < H$, the model uses grouped-query attention (GQA). |
| `head_dim` | `int` | width of a single attention head $h$. Note that $H \cdot h$ does **not** necessarily equal $d$. |
| `mlp_type` | `str` | `"swiglu"` or `"gelu"` |
| `d_ff` | `int` | hidden width of the MLP $f$ |
| `tie_embeddings` | `bool` | whether the output head shares its weight matrix with the token embedding table |
| `attn_bias` | `bool` | whether the four attention projections include bias vectors |
| `mlp_bias` | `bool` | whether the MLP linear layers include bias vectors |
| `norm_bias` | `bool` | whether each norm layer includes a bias vector in addition to its gain (LayerNorm when `True`, RMSNorm when `False`) |

A linear layer mapping `n_in` input features to `n_out` output features has a weight matrix of `n_in * n_out` parameters. If bias is enabled, it adds `n_out` more parameters.

The model consists of five components:

- **`embedding`**: One lookup table of shape `(vocab_size, d_model)` with $V d$ parameters.
- **`attention`** (per block): Four linear projections. The query projection $W_q$ maps $d \to H h$. The key and value projections $W_k$ and $W_v$ each map $d \to H_{kv} h$. The output projection $W_o$ maps $H h \to d$. Together, their weights total $2\,d\,H h + 2\,d\,H_{kv} h$ parameters. If `attn_bias` is `True`, the four bias vectors add $H h + 2 H_{kv} h + d$ parameters.
- **`mlp`** (per block):
  - For `"swiglu"`, there are three linear layers: a gate projection ($d \to f$), an up projection ($d \to f$), and a down projection ($f \to d$). Their weights total $3\,d\,f$ parameters, and if `mlp_bias` is `True`, their biases add $2f + d$ parameters. Every gated MLP, including Gemma's GELU-gated variant, uses this three-matrix layout.
  - For `"gelu"`, there are two linear layers: an up projection ($d \to f$) and a down projection ($f \to d$). Their weights total $2\,d\,f$ parameters, and if `mlp_bias` is `True`, their biases add $f + d$ parameters.
- **`norm`**: Each block has two normalization layers (one before attention and one before the MLP), and the model adds one final normalization layer before the output head, giving $2L + 1$ norms in total. Each norm has a gain vector of $d$ parameters, plus a bias vector of $d$ parameters if `norm_bias` is `True`.
- **`head`**: The output projection maps $d \to V$ without bias. If `tie_embeddings` is `True`, it shares the embedding table and counts as `0`. Otherwise, it has $V d$ parameters.

Return a dictionary with the exact keys `"embedding"`, `"attention"`, `"mlp"`, `"norm"`, `"head"`, and `"total"`. The `"attention"`, `"mlp"`, and `"norm"` entries are totals across all $L$ blocks (including the final norm for `"norm"`). `"total"` is the sum of the other five entries. All values must be Python `int`s, and `cfg` must not be modified.

## Example

Consider IBM's Granite 4.2 3B (`ibm-granite/granite-4.2-3b`). It has 40 blocks of width 2560, 40 query heads sharing 8 KV heads of width 64, a SwiGLU MLP of width 8192, no biases, RMSNorm, and an output head that is **not** tied to the embedding.

```python
granite_3b = {
    "vocab_size": 100352, "d_model": 2560, "n_layers": 40,
    "n_heads": 40, "n_kv_heads": 8, "head_dim": 64,
    "mlp_type": "swiglu", "d_ff": 8192, "tie_embeddings": False,
    "attn_bias": False, "mlp_bias": False, "norm_bias": False,
}
count_params(granite_3b)
# {"embedding": 256901120, "attention": 629145600, "mlp": 2516582400,
#  "norm": 207360, "head": 256901120, "total": 3659737600}
```

| component | arithmetic | parameters |
|---|---|---|
| embedding | $100352 \cdot 2560$ | 256,901,120 |
| attention | $40 \cdot (2 \cdot 2560 \cdot 2560 + 2 \cdot 2560 \cdot 512)$ | 629,145,600 |
| mlp | $40 \cdot 3 \cdot 2560 \cdot 8192$ | 2,516,582,400 |
| norm | $(2 \cdot 40 + 1) \cdot 2560$ | 207,360 |
| head | untied, $100352 \cdot 2560$ | 256,901,120 |
| **total** | | **3,659,737,600** |

Here $H h = 40 \cdot 64 = 2560$ and $H_{kv} h = 8 \cdot 64 = 512$, so the key and value projections are one-fifth the size of the query projection. The total matches the exact parameter count Hugging Face reports for the checkpoint.

Its smaller sibling, Granite 4.0 350M (`ibm-granite/granite-4.0-350m`), ties the output head to the embedding:

```python
granite_350m = {
    "vocab_size": 100352, "d_model": 1024, "n_layers": 28,
    "n_heads": 16, "n_kv_heads": 4, "head_dim": 64,
    "mlp_type": "swiglu", "d_ff": 2048, "tie_embeddings": True,
    "attn_bias": False, "mlp_bias": False, "norm_bias": False,
}
count_params(granite_350m)
# {"embedding": 102760448, "attention": 73400320, "mlp": 176160768,
#  "norm": 58368, "head": 0, "total": 352379904}
```

This again matches the published count of 352,379,904. Counting the tied head a second time would give 455,140,352, which is 29 percent too high.

Finally, here is a small configuration with every bias enabled so you can verify the bias terms by hand:

```python
tiny = {
    "vocab_size": 10, "d_model": 4, "n_layers": 2,
    "n_heads": 2, "n_kv_heads": 1, "head_dim": 3,
    "mlp_type": "gelu", "d_ff": 8, "tie_embeddings": False,
    "attn_bias": True, "mlp_bias": True, "norm_bias": True,
}
count_params(tiny)
# {"embedding": 40, "attention": 176, "mlp": 152, "norm": 40, "head": 40, "total": 448}
```

In each block (where $H h = 6$ and $H_{kv} h = 3$), attention has $2 \cdot 4 \cdot 6 + 2 \cdot 4 \cdot 3 = 72$ weights and $6 + 2 \cdot 3 + 4 = 16$ biases, giving $2 \cdot 88 = 176$ across 2 blocks. Each MLP has $2 \cdot 4 \cdot 8 = 64$ weights and $8 + 4 = 12$ biases, giving $2 \cdot 76 = 152$ across 2 blocks. There are $2 \cdot 2 + 1 = 5$ norms with $4 + 4 = 8$ parameters each, giving 40.

## Constraints

- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= n_heads <= 256`, and `n_kv_heads` divides `n_heads`.
- `1 <= head_dim <= 1024`, and `d_model`, `d_ff`, and `vocab_size` are positive integers at most 1,000,000.
- `mlp_type` is always `"swiglu"` or `"gelu"`.
- Counts are compared exactly. Use integer arithmetic, as totals exceed $2^{32}$.
- Components not listed above (such as learned position embeddings, QK-norm gains, adapters, and quantization scales) are out of scope and count as `0`.

## Why this matters

The parameter count is the starting point for every capacity plan. Multiplying it by the bytes per weight gives the memory a model occupies before the first request arrives, and it sets how many bytes must stream from memory for every generated token.

Counting from `config.json` is also the fastest sanity check when porting a model. If your formula disagrees with the checkpoint's parameter count, you immediately catch an architecture mismatch (such as an untied head or a wrong KV projection width) before running a single forward pass. Published numbers can also mislead. A checkpoint file that stores a tied matrix under two names advertises up to 26 percent more parameters than the model actually has.
