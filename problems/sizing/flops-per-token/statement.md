Count the floating-point operations a decoder-only transformer spends on a prompt
(prefill) and on each generated token (decode with a KV cache), from its
configuration. A FLOP count is only meaningful under a stated convention, so the
work is applying one convention consistently: which products are counted, over how
many positions, and with which head count.

## Specification

```python
def prefill_flops(cfg: dict, t: int) -> int: ...
def decode_flops(cfg: dict, context: int) -> int: ...
```

**The convention.** Multiplying an $m \times n$ matrix by an $n \times p$ matrix costs
$2\,m\,n\,p$ FLOPs (one multiplication and one addition per term). Only the matrix
products listed below are counted. Everything else is free: the embedding lookup,
norms, activation functions, the elementwise product in a gated MLP, rotary position
embeddings, the softmax, masking, residual and bias additions, and sampling.

**`cfg`** is a dict. The functions use these keys and must ignore any others:

| key | meaning |
|---|---|
| `vocab_size` | number of tokens $V$ |
| `d_model` | width of the residual stream $d$ |
| `n_layers` | number of blocks $L$ |
| `n_heads` | number of query heads $H$ |
| `n_kv_heads` | number of key/value heads $H_{kv}$ (grouped-query attention when smaller than $H$) |
| `head_dim` | width of one head $h$; $H h$ need not equal $d$ |
| `mlp_type` | `"swiglu"` (three matrices: gate, up, down) or `"gelu"` (two: up, down) |
| `d_ff` | hidden width of the MLP $f$ |

**Counted per block, for one position** (a $1 \times d$ row vector times a weight
matrix):

| product | shape of the weight | FLOPs |
|---|---|---|
| query projection | $d \times H h$ | $2\,d\,H h$ |
| key projection | $d \times H_{kv} h$ | $2\,d\,H_{kv} h$ |
| value projection | $d \times H_{kv} h$ | $2\,d\,H_{kv} h$ |
| output projection | $H h \times d$ | $2\,H h\,d$ |
| MLP, each of its 3 (`"swiglu"`) or 2 (`"gelu"`) matrices | $d \times f$ or $f \times d$ | $2\,d\,f$ each |

Call their sum $P$, the per-position cost of one block.

**Counted per block, for attention.** Every one of the $H$ query heads does two
products per query, over the $n$ positions it attends to:

- *scores*: the query ($1 \times h$) times the transposed keys ($h \times n$),
  $2\,h\,n$ FLOPs;
- *mixing*: the attention weights ($1 \times n$) times the values ($n \times h$),
  $2\,n\,h$ FLOPs.

Grouped-query attention shares each key/value head among $H / H_{kv}$ query heads,
but each query head still computes its own scores and its own mixture. One query
therefore costs $A(n) = H \cdot 4\,h\,n$ per block.

**Counted once: the head.** Projecting one hidden state onto the vocabulary is a
$1 \times d$ by $d \times V$ product, $2\,d\,V$ FLOPs. Only one position's logits are
needed to pick the next token, so the head is charged once per call of either
function.

**`prefill_flops(cfg, t)`**: the prompt's `t` positions go through every block in one
pass. Attention is computed densely: all `t` queries against all `t` keys, with the
causally masked half computed and then discarded, as engines do. The head is applied
to the last position only.

$$
\text{prefill}(t) = L \cdot \big(t \cdot P + t \cdot A(t)\big) + 2\,d\,V
$$

**`decode_flops(cfg, context)`**: one new position goes through every block. Its keys
and values are appended to the cache, and its single query per head attends over
`context` positions: every cached token and the new token itself.

$$
\text{decode}(\text{context}) = L \cdot \big(P + A(\text{context})\big) + 2\,d\,V
$$

Both functions return a Python int and do not modify `cfg`.

## Example

Granite 4.2 3B (`ibm-granite/granite-4.2-3b`): 40 blocks of width 2560, 40 query heads
over 8 KV heads of width 64, SwiGLU of width 8192, vocabulary 100,352.

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

Decode at a context of 4096: $40 \cdot (157286400 + 41943040) + 513802240 =
8{,}482{,}979{,}840$, about 8.5 GFLOP for one token.

Prefill of 4096 tokens: $40 \cdot 4096 \cdot (157286400 + 41943040) + 513802240 =
32{,}642{,}265{,}251{,}840$, about 32.6 TFLOP. It is not 4096 decode steps: the head
runs once instead of 4096 times, and every query pays for the full 4096 keys.

A configuration small enough to do by hand:

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

$P = 2 \cdot 4 \cdot (6 + 3 + 3 + 6) + 2 \cdot 2 \cdot 4 \cdot 8 = 144 + 128 = 272$,
$A(5) = 2 \cdot 4 \cdot 3 \cdot 5 = 120$, and the head costs $2 \cdot 4 \cdot 10 = 80$.
Decode: $2 \cdot (272 + 120) + 80 = 864$. Prefill: $2 \cdot 5 \cdot (272 + 120) + 80 =
4000$.

## Constraints

- `1 <= t <= 2,000,000` and `1 <= context <= 2,000,000`.
- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= n_heads <= 256`, `n_kv_heads` divides
  `n_heads`, `1 <= head_dim <= 1024`; `d_model`, `d_ff`, and `vocab_size` are at most
  1,000,000. `mlp_type` is always `"swiglu"` or `"gelu"`.
- Results are compared exactly. Use integer arithmetic: prefill counts for long
  prompts pass $2^{53}$ (about $9 \cdot 10^{15}$), beyond which a float no longer
  holds every integer.

## Why this matters

FLOPs divided by the rate a processor sustains is the compute-bound time, and the two
functions give the two halves of every latency budget: `prefill_flops` bounds time to
first token, `decode_flops` bounds time per output token. The shape of the formulas
tells you where an optimization can pay. Per token, the weight products are constant
and the attention term grows with the context, so short contexts are dominated by the
MLP and very long ones by attention; the KV cache is what keeps decode linear in the
context instead of quadratic; and prefill is thousands of positions through the same
matrices, which is why it saturates a GPU while single-stream decode leaves it waiting
for memory.
