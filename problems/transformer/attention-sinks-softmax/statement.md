Implement multi-head attention with learned sinks, the attention variant used by gpt-oss. Each head owns one extra learned logit that participates in the softmax alongside the key scores without carrying a value vector, allowing a query to place probability mass on the sink instead of on any real token.

![This problem is attention, with one extra learned logit per head inside its softmax.](assets/map.svg)

## Specification

```python
def attention_with_sinks(q, k, v, mask, sinks): ...
```

The inputs represent a batch of `B` items with `H` attention heads and head width `d`. The tensor `q` holds `T` query vectors per head, while `k` and `v` hold `S` key and value vectors per head (`S` need not equal `T`). The boolean `mask` is shared across all batch items and heads, with `mask[t, s]` set to `True` when query position `t` may attend to key position `s`. A row of `mask` may be all `False`. The vector `sinks` holds one scalar sink logit per head. An entry of `sinks` may be `-inf`, in which case every row of `mask` is guaranteed to contain at least one `True`.

For batch item $b$, head $h$, query position $t$, and key position $s$, compute the scaled dot-product scores as in standard attention:

$$
\text{score}_{b,h,t,s} = \frac{q_{b,h,t} \cdot k_{b,h,s}}{\sqrt{d}} .
$$

Let $\sigma_h = \text{sinks}[h]$ be the **sink logit** of head $h$. This single scalar is shared across all batch items and query positions for that head. Each query row takes a softmax over all unmasked key positions **and the sink**:

$$
Z_{b,h,t} = e^{\sigma_h} + \sum_{s' :\, \text{mask}[t, s']} e^{\text{score}_{b,h,t,s'}},
\qquad
a_{b,h,t,s} = \begin{cases}
\dfrac{e^{\text{score}_{b,h,t,s}}}{Z_{b,h,t}} & \text{mask}[t, s], \\[2ex]
0 & \text{otherwise.}
\end{cases}
$$

Discard the sink's probability share, $e^{\sigma_h} / Z_{b,h,t}$. Compute the output `y` as the weighted sum of the real key values only:

$$
y_{b,h,t} = \sum_{s} a_{b,h,t,s}\; v_{b,h,s} .
$$

Keep three properties of the sink in mind:

- **The sink is a logit, not a score.** $\sigma_h$ enters the softmax directly and is not divided by $\sqrt{d}$.
- **The sink is never masked.** `mask` applies only to key positions. If a row of `mask` is all `False`, all probability goes to the sink and $y_{b,h,t} = 0$.
- **The key weights are not renormalized.** For a finite sink, $\sum_s a_{b,h,t,s} = 1 - e^{\sigma_h}/Z_{b,h,t} < 1$. As $\sigma_h \to -\infty$, this sum approaches 1, so a head with `sinks[h] = -inf` behaves identically to standard attention.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call a built-in attention function or layer (such as `scaled_dot_product_attention`, `dot_product_attention`, or `MultiheadAttention`).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, H, T, d)` | float32 |
| `k`, `v` | `(B, H, S, d)` | float32 |
| `mask` | `(T, S)` | bool |
| `sinks` | `(H,)` | float32 |
| result | `(B, H, T, d)` | float32 |

## Example

Consider one batch item, three heads with identical queries, keys, and values, one query position, two unmasked key positions, and head width $d = 4$ (so $\sqrt{d} = 2$):

```python
# every head h:
q[0, h] = [[2.0, 0.0, 0.0, 0.0]]  # shape (1, 4)
k[0, h] = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]  # shape (2, 4)
v[0, h] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]  # shape (2, 4)
mask = [[True, True]]
sinks = [0.0, 2.0, float("-inf")]

attention_with_sinks(q, k, v, mask, sinks)
# [[[[0.576117, 0.211942, 0.0, 0.0]],     head 0, sink 0
#   [[0.244728, 0.090031, 0.0, 0.0]],     head 1, sink 2
#   [[0.731059, 0.268941, 0.0, 0.0]]]]    head 2, no sink
```

Every head has scaled scores $[2 \cdot 1 / 2,\; 0] = [1, 0]$. Because the two value vectors are the first two unit basis vectors, the first two coordinates of the output equal the attention weights directly.

- **Head 0** ($\sigma_0 = 0$): The softmax over $[1, 0, 0]$ is $[0.5761, 0.2119, 0.2119]$. The two keys receive the first two weights, which sum to $0.788$, and the sink absorbs the remaining $0.2119$.
- **Head 1** ($\sigma_1 = 2$): The softmax over $[1, 0, 2]$ is $[0.2447, 0.0900, 0.6652]$. Because the sink logit exceeds both key scores, it absorbs two-thirds of the probability mass.
- **Head 2** ($\sigma_2 = -\infty$): Since $e^{-\infty} = 0$, the weights match the standard softmax of $[1, 0]$.

## Constraints

- Results are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any NaN fails.
- $1 \le B \le 4$, $1 \le H, T, S \le 8$, and $1 \le d \le 16$. `H` may equal `T`.
- Scores satisfy $|\text{score}| \le 500$, and finite sink logits lie in $[-100, 100]$.
- No Python loops over batch items, heads, or positions are needed.

## Why this matters

Standard softmax forces every query to spend 100% of its attention weight on the available keys. When no token is relevant, trained transformers work around this constraint by dumping excess weight onto an arbitrary token, usually the first token in the sequence. That workaround is why evicting the beginning-of-sequence token from a sliding window or truncated KV cache degrades model quality.

A learned sink gives each head an explicit way to attend to nothing. For example, gpt-oss stores one sink logit per head in every attention layer of its checkpoint. An inference engine that ignores those sinks, renormalizes the key weights, or divides sinks by $\sqrt{d}$ will still run without errors and produce fluent text, but from the wrong model. Because the sink changes the softmax denominator, fused attention kernels must support it natively (as in MLX's `scaled_dot_product_attention(..., sinks=...)`).
