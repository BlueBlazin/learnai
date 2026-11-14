## The idea

A forward pass is a sequence of matrix products, and under the convention each one
costs twice the number of scalar multiplications. There are three kinds.

**Weight products.** A position's hidden state is multiplied by every weight matrix
of every block: four projections and two or three MLP matrices. A $1 \times n$ row
times an $n \times p$ matrix costs $2np$, which is twice the number of weights in the
matrix. So the per-position cost of the blocks is twice their matrix parameters:

$$
P = 2 \cdot \big(\underbrace{d\,Hh}_{W_q} + \underbrace{2\,d\,H_{kv}h}_{W_k,\,W_v} +
\underbrace{Hh\,d}_{W_o} + \underbrace{k\,d\,f}_{\text{MLP},\ k \in \{2, 3\}}\big).
$$

This is the origin of the rule of thumb "a forward pass costs about 2 FLOPs per
parameter per token". It holds for matrices, not for the embedding table, which is
indexed and never multiplied.

**Attention products.** These involve no weights, only activations, so they do not
show up in a parameter count. A query head multiplies its query by $n$ keys
($2hn$ FLOPs) and then its $n$ attention weights by $n$ values ($2nh$). All $H$ query
heads do this. Grouped-query attention makes several of them read the *same* keys and
values, which shrinks the cache and the K/V projections, but not this work: the
number of score rows is $H$, not $H_{kv}$.

**The head.** $2dV$, once: logits are needed only for the position whose successor is
being chosen.

**Prefill versus decode.** In a prefill of $t$ tokens there are $t$ positions and $t$
queries, each against $t$ keys (engines compute the dense $t \times t$ product and
mask it). With a KV cache, a decode step has one position and one query against
`context` keys; the keys and values of earlier tokens are read, not recomputed.

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

For Granite 4.2 3B (configuration from
`https://huggingface.co/ibm-granite/granite-4.2-3b/raw/main/config.json`, read on
2026-11-14):

| | weight products | attention | head | total |
|---|---|---|---|---|
| decode, context 1,000 | 6.29 GFLOP | 0.41 GFLOP (6%) | 0.51 GFLOP | 7.21 GFLOP |
| decode, context 4,096 | 6.29 GFLOP | 1.68 GFLOP (20%) | 0.51 GFLOP | 8.48 GFLOP |
| decode, context 131,072 | 6.29 GFLOP | 53.7 GFLOP (89%) | 0.51 GFLOP | 60.5 GFLOP |
| prefill, 4,096 tokens | 25.8 TFLOP | 6.87 TFLOP (21%) | 0.51 GFLOP | 32.6 TFLOP |

- The weight products plus the head are $6{,}805{,}258{,}240$ FLOPs, exactly twice
  the model's matrix parameters (3,659,737,600 total, minus 207,360 norm gains and
  the 256,901,120-entry embedding table). `param-count` and this problem are the same
  bookkeeping.
- Attention overtakes the weight products when $4Hh \cdot c = P$, at a context of
  15,360 tokens for this model. Below that, per-token cost is nearly flat; far above
  it, cost per token is proportional to the context.
- A dense prefill of $t$ tokens costs more than the $t$ cached decode steps that would
  process the same tokens one at a time ($4Hh\,t^2$ against $4Hh\,t(t+1)/2$ for
  attention), because it computes the masked half of the score matrix. It is still
  far faster in practice, because one large product uses the hardware much better
  than $t$ small ones.
- Without a cache, generating token $c$ costs a prefill of $c$ tokens. For 1,000 new
  tokens after a 1,000-token prompt that is $1.04 \cdot 10^{19}$ FLOPs against
  $1.41 \cdot 10^{16}$ with the cache, about 740 times more.

## Common mistakes

- **Charging decode the quadratic term.** Using `context * context` in
  `decode_flops` describes a model with no KV cache. It is right at context 1 and
  wrong by a factor that grows with the context.
- **Charging the head at every position.** The head is the largest single matrix
  ($2dV$ = 0.51 GFLOP here, more than three blocks). Multiplying it by `t` in prefill,
  or by `context` in decode, inflates a 4096-token prefill by 6 percent and a decode
  step at that context by a factor of 250.
- **Counting scores for KV heads.** $H_{kv}$ in the attention term undercounts by the
  grouping factor (5x for Granite). GQA shares keys and values between query heads;
  it does not remove query heads.
- **Counting K and V projections at full width.** The mirror image: $W_k$ and $W_v$
  really are narrower under GQA, $d \times H_{kv}h$.
- **One FLOP per multiply.** $mnp$ instead of $2mnp$ counts multiply-accumulates
  (MACs). Both conventions are in use, and they differ by exactly 2, so always say
  which one a number is in.
- **Forgetting the mixing product.** $QK^\top$ is half of attention; the weighted sum
  of the values costs the same again.
- **Two matrices for a gated MLP.** SwiGLU has three; the MLP is 80 percent of $P$
  here, so the error is large.
- **Excluding the current token from `context`.** The new token attends to itself:
  its own key and value are in the cache when its query runs.

## Complexity

$O(1)$ arithmetic. The quantities themselves are $O(L\,t\,(P + Hh\,t))$ for prefill
and $O(L\,(P + Hh\,c))$ per decoded token. Use integers: the example's
$3.3 \cdot 10^{13}$ is below $2^{53}$, but a prefill of a million tokens is not.

## In the wild

- **Time to first token.** A GPU that sustained $10^{14}$ FLOP/s on this workload
  would prefill 4,096 tokens of Granite 4.2 3B in about a third of a second. Prefill
  is the regime where such an estimate is close, because large dense products keep
  the arithmetic units busy.
- **Decode is not compute-bound at small batch.** 8.5 GFLOP at the same rate would be
  85 microseconds per token; no single stream decodes that fast, because each step
  must first pull every weight through memory. That bound is the subject of
  `memory-bandwidth-bound`, and the gap between the two is the headroom that batching
  fills.
- **Conventions differ between papers.** The $2N$ FLOPs-per-token estimate (Kaplan et
  al., 2020) uses this convention for the weights; some profilers count MACs, some
  count the masked half of attention and some do not, and many ignore attention
  entirely, which is harmless at 1,000 tokens and wrong by 9x at 131,072.
