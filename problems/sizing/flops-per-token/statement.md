Count the floating-point operations a decoder-only transformer spends on a prompt (prefill) and on each generated token (decode with a KV cache) from its configuration. Because FLOP counts depend on which operations are included, both functions follow one explicit matrix-multiplication convention.

## Specification

```python
def prefill_flops(cfg: dict, t: int) -> int: ...
def decode_flops(cfg: dict, context: int) -> int: ...
```

**FLOP convention.** Multiplying an $m \times n$ matrix by an $n \times p$ matrix costs $2\,m\,n\,p$ FLOPs (one multiplication and one addition per inner-product term). Only the matrix products listed below are counted. All other operations are excluded, including the embedding lookup, norms, activation functions, the elementwise product in a gated MLP, rotary position embeddings, softmax, masking, residual and bias additions, and sampling.

**Configuration.** `cfg` is a dictionary containing these keys. Both functions must ignore any additional keys.

| key | meaning |
|---|---|
| `vocab_size` | vocabulary size $V$ |
| `d_model` | width of the residual stream $d$ |
| `n_layers` | number of transformer blocks $L$ |
| `n_heads` | number of query heads $H$ |
| `n_kv_heads` | number of key/value heads $H_{kv}$ (grouped-query attention when $H_{kv} < H$) |
| `head_dim` | width of one head $h$ ($H h$ need not equal $d$) |
| `mlp_type` | `"swiglu"` (three matrices: gate, up, down) or `"gelu"` (two matrices: up, down) |
| `d_ff` | hidden width of the MLP $f$ |

**Per-position weight products (per block).** For a single token position, each weight matrix in a block multiplies a 1-row activation vector by that matrix.

| product | weight shape | FLOPs |
|---|---|---|
| query projection | $d \times H h$ | $2\,d\,H h$ |
| key projection | $d \times H_{kv} h$ | $2\,d\,H_{kv} h$ |
| value projection | $d \times H_{kv} h$ | $2\,d\,H_{kv} h$ |
| output projection | $H h \times d$ | $2\,H h\,d$ |
| MLP, each of its 3 (`"swiglu"`) or 2 (`"gelu"`) matrices | $d \times f$ or $f \times d$ | $2\,d\,f$ each |

Let $P$ denote the sum of these products for one position in one block.

**Attention products (per block).** When one query attends over $n$ key/value positions, each of the $H$ query heads performs two matrix products.

- **Scores.** Multiplying the query ($1 \times h$) by the transposed keys ($h \times n$) costs $2\,h\,n$ FLOPs.
- **Value mixing.** Multiplying the attention weights ($1 \times n$) by the values ($n \times h$) costs $2\,n\,h$ FLOPs.

Grouped-query attention shares each key/value head across $H / H_{kv}$ query heads, but every query head still computes its own scores and its own weighted sum of values. One query attending over $n$ positions therefore costs $A(n) = H \cdot 4\,h\,n$ FLOPs per block.

**Output head (counted once).** Projecting one hidden state onto the vocabulary multiplies a $1 \times d$ vector by a $d \times V$ matrix, which costs $2\,d\,V$ FLOPs. Only one position's logits are needed to select the next token, so the output head is counted once per call to either function.

**`prefill_flops(cfg, t)`.** All `t` prompt positions pass through every block in one forward pass. Attention is computed densely across all `t` queries and all `t` keys, including the causally masked half that is later discarded. The output head runs on the last position only.

$$
\text{prefill}(t) = L \cdot \big(t \cdot P + t \cdot A(t)\big) + 2\,d\,V
$$

**`decode_flops(cfg, context)`.** One new token position passes through every block. Its key and value vectors are appended to the KV cache, and each query head attends over all `context` positions (all previously cached tokens plus the new token itself).

$$
\text{decode}(\text{context}) = L \cdot \big(P + A(\text{context})\big) + 2\,d\,V
$$

Both functions return a Python `int` and do not modify `cfg`.

## Example

Granite 4.2 3B (`ibm-granite/granite-4.2-3b`) has 40 blocks of width 2560, 40 query heads over 8 KV heads of width 64, a SwiGLU MLP of width 8192, and a vocabulary of 100,352 tokens.

```python
granite_3b = {
    "vocab_size": 100352, "d_model": 2560, "n_layers": 40,
    "n_heads": 40, "n_kv_heads": 8, "head_dim": 64,
    "mlp_type": "swiglu", "d_ff": 8192,
}
decode_flops(granite_3b, 4096)    # 8482979840
prefill_flops(granite_3b, 4096)   # 32642265251840
```

Here $H h = 2560$ and $H_{kv} h = 512$.

| term | arithmetic | FLOPs |
|---|---|---|
| projections, per position per block | $2 \cdot 2560 \cdot (2560 + 512 + 512 + 2560)$ | 31,457,280 |
| MLP, per position per block | $3 \cdot 2 \cdot 2560 \cdot 8192$ | 125,829,120 |
| $P$ | | 157,286,400 |
| $A(4096)$, per query per block | $40 \cdot 4 \cdot 64 \cdot 4096$ | 41,943,040 |
| head | $2 \cdot 2560 \cdot 100352$ | 513,802,240 |

For decode at a context of 4096, the total is $40 \cdot (157286400 + 41943040) + 513802240 = 8{,}482{,}979{,}840$ FLOPs (about 8.5 GFLOP per token).

For prefill of 4096 tokens, the total is $40 \cdot 4096 \cdot (157286400 + 41943040) + 513802240 = 32{,}642{,}265{,}251{,}840$ FLOPs (about 32.6 TFLOP). This is not equal to 4096 decode steps because the output head runs once instead of 4096 times, and every query pays for all 4096 keys.

A smaller configuration illustrates the arithmetic by hand.

```python
tiny = {
    "vocab_size": 10, "d_model": 4, "n_layers": 2,
    "n_heads": 2, "n_kv_heads": 1, "head_dim": 3,
    "mlp_type": "gelu", "d_ff": 8,
}
decode_flops(tiny, 5)     # 864
prefill_flops(tiny, 5)    # 4000
prefill_flops(tiny, 1)    # 672, the same as decode_flops(tiny, 1)
```

Here $P = 2 \cdot 4 \cdot (6 + 3 + 3 + 6) + 2 \cdot 2 \cdot 4 \cdot 8 = 144 + 128 = 272$, $A(5) = 2 \cdot 4 \cdot 3 \cdot 5 = 120$, and the output head costs $2 \cdot 4 \cdot 10 = 80$. Decode costs $2 \cdot (272 + 120) + 80 = 864$. Prefill costs $2 \cdot 5 \cdot (272 + 120) + 80 = 4000$.

## Constraints

- `1 <= t <= 2,000,000` and `1 <= context <= 2,000,000`.
- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= n_heads <= 256`, `n_kv_heads` divides `n_heads`, and `1 <= head_dim <= 1024`.
- `d_model`, `d_ff`, and `vocab_size` are positive integers at most `1,000,000`.
- `mlp_type` is always `"swiglu"` or `"gelu"`.
- Results are compared exactly. Use integer arithmetic because prefill counts for long prompts exceed $2^{53}$ (about $9 \cdot 10^{15}$), beyond which floating-point numbers cannot represent every integer exactly.

## Why this matters

Dividing FLOPs by a processor's sustained compute rate gives the compute-bound latency. `prefill_flops` bounds time to first token, while `decode_flops` bounds time per output token.

The two formulas also show where optimization effort pays off. Per token, weight products are constant while attention grows with context length, so MLPs dominate at short contexts and attention dominates at very long contexts. The KV cache keeps decode linear in context length instead of quadratic. Meanwhile, prefill processes many positions through the same weight matrices at once, saturating GPU compute while single-stream decode waits on memory bandwidth.
