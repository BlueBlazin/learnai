Compute how many parameters a decoder-only transformer has from its configuration
alone, broken down by component. The arithmetic is multiplication; the difficulty is
the conventions, where each wrong guess (a tied head, grouped-query attention, a gated
MLP) moves the answer by 10 to 40 percent.

## Specification

```python
def count_params(cfg: dict) -> dict[str, int]: ...
```

`cfg` describes a pre-norm decoder with rotary position embeddings (which have no
parameters). It is a dict with exactly these keys:

| key | type | meaning |
|---|---|---|
| `vocab_size` | int | number of tokens $V$ |
| `d_model` | int | width of the residual stream $d$ |
| `n_layers` | int | number of transformer blocks $L$ |
| `n_heads` | int | number of query heads $H$ |
| `n_kv_heads` | int | number of key and value heads $H_{kv}$; it divides `n_heads`. Fewer KV heads than query heads is grouped-query attention (GQA) |
| `head_dim` | int | width of one head $h$. $H \cdot h$ is **not** necessarily equal to $d$ |
| `mlp_type` | str | `"swiglu"` or `"gelu"` |
| `d_ff` | int | hidden width of the MLP $f$ |
| `tie_embeddings` | bool | the output head shares its matrix with the token embedding |
| `attn_bias` | bool | the four attention projections have bias vectors |
| `mlp_bias` | bool | the MLP's linear layers have bias vectors |
| `norm_bias` | bool | each norm has a bias in addition to its gain (LayerNorm); `False` is RMSNorm, gain only |

A linear layer from `n_in` features to `n_out` features has a weight matrix of
`n_in * n_out` parameters and, if it has a bias, `n_out` more. The components are:

- **`embedding`**: one `vocab_size x d_model` table, $V d$.
- **`attention`**, per block: four linear layers.
  $W_q$ maps $d \to H h$, $W_k$ and $W_v$ each map $d \to H_{kv} h$, and $W_o$ maps
  $H h \to d$. Weights: $2\,d\,H h + 2\,d\,H_{kv} h$. Biases, if `attn_bias`:
  $H h + 2 H_{kv} h + d$.
- **`mlp`**, per block:
  - `"swiglu"`: three linear layers, gate and up ($d \to f$ each) and down
    ($f \to d$). Weights $3\,d\,f$; biases, if `mlp_bias`, $2f + d$. Every gated MLP
    has this shape, including the GELU-gated one in Gemma.
  - `"gelu"`: two linear layers, up ($d \to f$) and down ($f \to d$). Weights
    $2\,d\,f$; biases, if `mlp_bias`, $f + d$.
- **`norm`**: two norms per block (before attention and before the MLP) and one final
  norm before the head, $2L + 1$ in total. Each has a gain vector of $d$ parameters,
  plus a bias vector of $d$ parameters if `norm_bias`.
- **`head`**: the output projection $d \to V$, without bias. If `tie_embeddings` is
  true it is the same tensor as the embedding table and counts **0**; otherwise $V d$.

The function returns a dict with exactly the keys `"embedding"`, `"attention"`,
`"mlp"`, `"norm"`, `"head"`, and `"total"`. `"attention"`, `"mlp"`, and `"norm"` are
totals over all $L$ blocks (and the final norm), and `"total"` is the sum of the other
five. All values are Python ints. The function does not modify `cfg`.

## Example

IBM's Granite 4.2 3B (`ibm-granite/granite-4.2-3b`): 40 blocks of width 2560, 40
query heads sharing 8 KV heads of width 64, a SwiGLU MLP of width 8192, no biases,
RMSNorm, and an output head that is **not** tied to the embedding.

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
| head | untied: $100352 \cdot 2560$ | 256,901,120 |
| **total** | | **3,659,737,600** |

Here $H h = 40 \cdot 64 = 2560$ and $H_{kv} h = 8 \cdot 64 = 512$: the key and value
projections are a fifth the size of the query projection. The total is exactly the
parameter count Hugging Face reports for the checkpoint.

Its small sibling Granite 4.0 350M (`ibm-granite/granite-4.0-350m`) ties the head:

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

Again the published count, 352,379,904. Counting the tied head a second time would
give 455,140,352, which is 29 percent too many.

A small configuration with every bias switched on, to check the bias terms by hand:

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

Attention per block: weights $2 \cdot 4 \cdot 6 + 2 \cdot 4 \cdot 3 = 72$, biases
$6 + 2 \cdot 3 + 4 = 16$, so $2 \cdot 88 = 176$. MLP per block: $2 \cdot 4 \cdot 8 = 64$
weights and $8 + 4 = 12$ biases, so $2 \cdot 76 = 152$. Norms: $5 \cdot (4 + 4) = 40$.

## Constraints

- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= n_heads <= 256`, and `n_kv_heads`
  divides `n_heads`; `1 <= head_dim <= 1024`; `d_model`, `d_ff`, and `vocab_size` are
  at most 1,000,000.
- `mlp_type` is always `"swiglu"` or `"gelu"`.
- Counts are compared exactly. Use integer arithmetic; totals exceed $2^{32}$.
- Everything not listed above is out of scope and counts zero: learned position
  embeddings, QK-norm gains, adapters, and quantization scales.

## Why this matters

The parameter count is the first number in every capacity plan: multiplied by the
bytes per weight it is the memory the model occupies before the first request arrives,
and it sets the bytes that have to move through memory for every generated token. It
is also the cheapest sanity check on a port: if your count from `config.json` does not
match the checkpoint's, you have misread the architecture (an untied head, a KV
projection of the wrong width) before running a single forward pass. And published
numbers are not always what they seem: a file that stores a tied matrix twice
advertises 26 percent more parameters than the model has.
