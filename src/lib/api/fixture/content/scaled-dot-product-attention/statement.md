Attention is the one operation every transformer spends most of its time in. Given queries $Q \in \mathbb{R}^{T_q \times d}$, keys $K \in \mathbb{R}^{T_k \times d}$, and values $V \in \mathbb{R}^{T_k \times d_v}$, it computes

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}} + M\right) V ,
$$

where $M$ is an additive mask whose entries are $0$ (attend) or $-\infty$ (do not attend). The scale $1/\sqrt{d}$ keeps the variance of the scores near $1$ regardless of the head dimension, so the softmax neither saturates nor flattens as $d$ grows.

Implement `attention(q, k, v, mask)` for batched, multi-head inputs.

## Interface

```python
def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    q: (B, H, Tq, d)    k: (B, H, Tk, d)    v: (B, H, Tk, dv)
    mask: broadcastable to (B, H, Tq, Tk), True where attention is allowed.
    Returns (B, H, Tq, dv).
    """
```

Note that the mask is **boolean** here, not additive; converting it is part of the problem. You may not call `scaled_dot_product_attention` or any `MultiheadAttention` module.

### Example 1

One head, two positions, identity values. With $d = 1$ the scale is $1$.

```python
q = torch.tensor([[[[1.0], [0.0]]]])      # (1, 1, 2, 1)
k = torch.tensor([[[[1.0], [2.0]]]])
v = torch.tensor([[[[10.0], [20.0]]]])
attention(q, k, v)
# tensor([[[[17.3106], [15.0000]]]])
```

The first query scores the keys $[1, 2]$, giving weights $[0.269, 0.731]$ and output $0.269 \cdot 10 + 0.731 \cdot 20 = 17.31$. The second query is $0$, so it scores both keys equally and returns the mean.

### Example 2

The same inputs with a causal mask. Position $0$ may only see itself.

```python
mask = torch.tensor([[True, False], [True, True]])
attention(q, k, v, mask)
# tensor([[[[10.0000], [15.0000]]]])
```

## Shapes at a glance

| Tensor | Shape | Meaning |
|---|---|---|
| `q` | $(B, H, T_q, d)$ | queries, one per output position |
| `k` | $(B, H, T_k, d)$ | keys, one per source position |
| `v` | $(B, H, T_k, d_v)$ | values aligned with keys |
| scores | $(B, H, T_q, T_k)$ | before and after softmax |
| output | $(B, H, T_q, d_v)$ | weighted mix of values |

> [!WARNING]
> Scale by $\sqrt{d}$ where $d$ is the **key** dimension, the last axis of `k`. Using $d_v$ passes the cases where $d = d_v$ and fails the rest.

## Constraints

- $1 \le B \le 8$, $1 \le H \le 16$, $1 \le T_q, T_k \le 512$, $d, d_v \in \{1, 16, 64, 128\}$.
- Every query row has at least one allowed key.
- Tolerance: absolute $10^{-5}$, relative $10^{-4}$.
