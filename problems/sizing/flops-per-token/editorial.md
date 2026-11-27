## The idea

A transformer forward pass consists of three categories of matrix multiplications. Under the $2mnp$ convention, each multiplication costs twice its number of scalar multiply-accumulates.

**Weight products.** Every token position's hidden state of width $d$ is multiplied by every weight matrix in each of the $L$ blocks, which includes four attention projections and either two (`"gelu"`) or three (`"swiglu"`) MLP matrices. Multiplying a $1 \times n$ row by an $n \times p$ weight matrix costs $2np$ FLOPs, which is twice the number of entries in the matrix. With $H$ query heads, $H_{kv}$ key/value heads, head width $h$, and MLP hidden width $f$, the per-position weight cost of one block is

$$
P = 2 \cdot \big(\underbrace{d\,Hh}_{W_q} + \underbrace{2\,d\,H_{kv}h}_{W_k,\,W_v} +
\underbrace{Hh\,d}_{W_o} + \underbrace{k\,d\,f}_{\text{MLP},\ k \in \{2, 3\}}\big).
$$

This is where the rule of thumb "a forward pass costs about 2 FLOPs per parameter per token" comes from. It applies to weight matrices, not to the token embedding table, which is indexed rather than multiplied.

**Attention products.** Attention multiplies activations by activations, so it does not appear in a parameter count. Each of the $H$ query heads multiplies its query by $n$ keys ($2hn$ FLOPs) and then multiplies its $n$ attention weights by $n$ values ($2nh$ FLOPs), for $4Hh\,n$ FLOPs per query per block. Grouped-query attention shares keys and values across query heads, which shrinks the KV cache and the $W_k$ and $W_v$ projections, but every query head still computes its own score row and value mixture. The factor is therefore $H$, not $H_{kv}$.

**Output head.** Projecting one hidden state onto the vocabulary of size $V$ costs $2dV$ FLOPs. It runs once per call because only one position's logits are needed to select the next token.

**Prefill versus decode.** In a prefill of $t$ tokens, all $t$ positions pass through the weight matrices and all $t$ queries attend densely over all $t$ keys before causal masking is applied. In a cached decode step at context length $c$, only one new position passes through the weights and its single query attends over all $c$ keys already in the cache (including its own). Earlier keys and values are read from the cache rather than recomputed.

$$
\text{prefill}(t) = L\,(tP + 4Hh\,t^2) + 2dV, \qquad
\text{decode}(c) = L\,(P + 4Hh\,c) + 2dV .
$$

## Reference solution (Python)

```python
def _per_position_flops(cfg):
    d = cfg["d_model"]
    q_width = cfg["n_heads"] * cfg["head_dim"]
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]
    projections = 2 * d * q_width + 2 * (2 * d * kv_width) + 2 * q_width * d
    n_mlp_matrices = 3 if cfg["mlp_type"] == "swiglu" else 2
    return projections + n_mlp_matrices * 2 * d * cfg["d_ff"]


def _attention_flops(cfg, n_queries, n_keys):
    scores = 2 * n_queries * cfg["head_dim"] * n_keys
    mixing = 2 * n_queries * n_keys * cfg["head_dim"]
    return cfg["n_heads"] * (scores + mixing)


def _head_flops(cfg):
    return 2 * cfg["d_model"] * cfg["vocab_size"]


def prefill_flops(cfg, t):
    per_layer = t * _per_position_flops(cfg) + _attention_flops(cfg, t, t)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)


def decode_flops(cfg, context):
    per_layer = _per_position_flops(cfg) + _attention_flops(cfg, 1, context)
    return cfg["n_layers"] * per_layer + _head_flops(cfg)
```

## Reading the numbers

For Granite 4.2 3B (configuration from `https://huggingface.co/ibm-granite/granite-4.2-3b/raw/main/config.json`, read on 2026-11-14):

| | weight products | attention | head | total |
|---|---|---|---|---|
| decode, context 1,000 | 6.29 GFLOP | 0.41 GFLOP (6%) | 0.51 GFLOP | 7.21 GFLOP |
| decode, context 4,096 | 6.29 GFLOP | 1.68 GFLOP (20%) | 0.51 GFLOP | 8.48 GFLOP |
| decode, context 131,072 | 6.29 GFLOP | 53.7 GFLOP (89%) | 0.51 GFLOP | 60.5 GFLOP |
| prefill, 4,096 tokens | 25.8 TFLOP | 6.87 TFLOP (21%) | 0.51 GFLOP | 32.6 TFLOP |

- The weight products plus the output head total $6{,}805{,}258{,}240$ FLOPs, which is exactly twice the model's matrix parameters (3,659,737,600 total parameters, minus 207,360 norm gains and the 256,901,120-entry embedding table).
- Attention overtakes weight products when $4Hh \cdot c = P$, which happens at a context of 15,360 tokens for this model. Below that point, per-token cost is nearly flat. Far above it, cost per token grows linearly with context length.
- A dense prefill of $t$ tokens performs more attention FLOPs than $t$ cached decode steps over the same tokens ($4Hh\,t^2$ versus $4Hh\,t(t+1)/2$) because it computes the masked half of the score matrix. Prefill is still much faster in practice because one large matrix multiplication uses hardware far more efficiently than $t$ small ones.
- Without a KV cache, generating token $c$ costs a full prefill of $c$ tokens. Generating 1,000 new tokens after a 1,000-token prompt costs $1.04 \cdot 10^{19}$ FLOPs without a cache versus $1.41 \cdot 10^{16}$ FLOPs with a cache, about 740 times more.

## Common mistakes

- **Charging decode the quadratic term.** Using `context * context` in `decode_flops` models decoding without a KV cache. It matches at context 1 and diverges by a factor that grows with `context`.
- **Charging the output head at every position.** The head is the largest single matrix ($2dV = 0.51$ GFLOP for Granite 4.2 3B, more than three blocks). Multiplying it by `t` in prefill or by `context` in decode inflates a 4096-token prefill by 6 percent and a decode step at that context by a factor of 250.
- **Counting attention scores with `n_kv_heads`.** Using $H_{kv}$ in the attention term undercounts by the grouping factor (5x for Granite 4.2 3B). GQA shares keys and values across query heads without removing query heads.
- **Counting K and V projections at full width.** Under GQA, $W_k$ and $W_v$ really are narrower, with shape $d \times H_{kv}h$ rather than $d \times Hh$.
- **Counting one FLOP per multiply.** Using $mnp$ instead of $2mnp$ counts multiply-accumulates (MACs). Both conventions exist and differ by a factor of 2, so the convention must always be explicit.
- **Forgetting the value-mixing product.** Computing $QK^\top$ is only half of attention. Multiplying the attention weights by $V$ costs the same number of FLOPs again.
- **Using two matrices for a gated MLP.** SwiGLU has three matrices (gate, up, down). Because the MLP accounts for 80 percent of $P$ in Granite 4.2 3B, dropping one matrix causes a large error.
- **Excluding the current token from `context`.** The new token appends its own key and value to the cache before its query runs, so it attends over all `context` positions including itself.

## Complexity

$O(1)$ arithmetic per call. The returned counts grow as $O(L\,t\,(P + Hh\,t))$ for prefill and $O(L\,(P + Hh\,c))$ per decoded token. Use integer arithmetic. The example's $3.3 \cdot 10^{13}$ fits in a 64-bit float, but a prefill of one million tokens exceeds $2^{53}$.

## In the wild

- **Time to first token.** A GPU sustaining $10^{14}$ FLOP/s on this workload prefills 4,096 tokens of Granite 4.2 3B in about a third of a second. Prefill estimates are close to measured latency because large dense matrix multiplications keep compute units saturated.
- **Decode is not compute-bound at small batch sizes.** At $10^{14}$ FLOP/s, 8.5 GFLOP would take 85 microseconds per token. No single stream decodes that fast because each step must load every model weight from memory. That memory-bandwidth ceiling is analyzed in `memory-bandwidth-bound`, and batching exists to close the gap between the two limits.
- **Conventions vary across papers and tools.** The $2N$ FLOPs-per-token estimate for an $N$-parameter model (Kaplan et al., 2020) uses this convention for weights. Some profilers count MACs instead of FLOPs, some exclude the masked half of attention, and many omit attention altogether, which is nearly exact at 1,000 tokens and off by 9x at 131,072 tokens.
