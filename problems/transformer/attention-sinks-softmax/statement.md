Implement multi-head attention with learned sinks, the variant used by gpt-oss: each
head owns one extra logit that takes part in the softmax next to the scores and
carries no value. A query can put probability on the sink instead of on any token, so
its weights over the real keys sum to less than 1. The arithmetic is one extra column;
the problem is that every way of getting that column slightly wrong still returns a
tensor of the right shape.

![This problem is attention, with one extra learned logit per head inside its softmax.](assets/map.svg)

## Specification

```python
def attention_with_sinks(q, k, v, mask, sinks): ...
```

For batch item $b$, head $h$, query position $t$, and key position $s$, with $d$ the
width of a head, the scores are those of ordinary attention:

$$
\text{score}_{b,h,t,s} = \frac{q_{b,h,t} \cdot k_{b,h,s}}{\sqrt{d}} .
$$

Head $h$ has a **sink logit** $\sigma_h$ = `sinks[h]`: one scalar, the same for every
batch item and every query position. The softmax of a row runs over the key
positions that the mask allows **and the sink**:

$$
Z_{b,h,t} = e^{\sigma_h} + \sum_{s' :\, \text{mask}[t, s']} e^{\text{score}_{b,h,t,s'}},
\qquad
a_{b,h,t,s} = \begin{cases}
\dfrac{e^{\text{score}_{b,h,t,s}}}{Z_{b,h,t}} & \text{mask}[t, s], \\[2ex]
0 & \text{otherwise.}
\end{cases}
$$

The sink's own share, $e^{\sigma_h} / Z_{b,h,t}$, is discarded. The output is the
weighted sum of the values of the real keys only:

$$
y_{b,h,t} = \sum_{s} a_{b,h,t,s}\; v_{b,h,s} .
$$

Three properties of the sink, each of which the tests check:

- **It is a logit, not a score.** $\sigma_h$ enters the softmax as it is. It is not
  divided by $\sqrt{d}$.
- **It is never masked.** The mask describes key positions; the sink is not one.
  Consequently a row whose mask is all `False` is well defined: all of its
  probability goes to the sink and $y_{b,h,t} = 0$.
- **The weights are not renormalized.** For a finite sink,
  $\sum_s a_{b,h,t,s} = 1 - e^{\sigma_h}/Z_{b,h,t} < 1$. As $\sigma_h \to -\infty$
  the sum goes to 1, and a head with `sinks[h] = -inf` is exactly ordinary attention.

Arguments:

- `q`: float32, shape `(B, H, T, d)`. `T` query positions.
- `k`, `v`: float32, shape `(B, H, S, d)`. `S` key positions; `S` need not equal `T`.
- `mask`: bool, shape `(T, S)`. `mask[t, s]` is `True` where query position `t` may
  attend to key position `s`. It is shared by all batch items and heads. A row may be
  all `False`.
- `sinks`: float32, shape `(H,)`. An entry may be `-inf`, but then every row of the
  mask contains a `True` (otherwise the row would have nothing to attend to).
- Returns `y`: float32, shape `(B, H, T, d)`.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call
a built-in attention function or layer (`scaled_dot_product_attention`,
`dot_product_attention`, `MultiheadAttention`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, H, T, d)` | float32 |
| `k`, `v` | `(B, H, S, d)` | float32 |
| `mask` | `(T, S)` | bool |
| `sinks` | `(H,)` | float32 |
| result | `(B, H, T, d)` | float32 |

## Example

One batch item, three heads with the same query, keys, and values, one query
position, two key positions, $d = 4$ (so $\sqrt{d} = 2$), nothing masked:

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

The scores are $[2 \cdot 1 / 2,\; 0] = [1, 0]$ in every head, and the values are unit
vectors, so the output shows the weights directly.

- Head 0: softmax over $[1, 0, \sigma = 0]$ is $[0.5761, 0.2119, 0.2119]$. The keys
  get the first two numbers, which sum to $0.788$; the last is the sink's share.
- Head 1: softmax over $[1, 0, 2]$ is $[0.2447, 0.0900, 0.6652]$. The sink logit is
  larger than both scores and takes two thirds of the probability.
- Head 2: $e^{-\infty} = 0$, and the weights are the ordinary softmax of $[1, 0]$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le B \le 4$; $1 \le H, T, S \le 8$; $1 \le d \le 16$. `H` may equal `T`.
- Scores satisfy $|\text{score}| \le 500$, and finite sink logits lie in
  $[-100, 100]$.
- No Python loops over batch items, heads, or positions are needed.

## Why this matters

A softmax forces every query to spend all of its attention somewhere. When no token
is relevant, trained models work around that by dumping the weight on an arbitrary
token, usually the first one, which is why the beginning-of-sequence token attracts
so much attention in ordinary transformers and why evicting it from a sliding window
or a truncated cache degrades them. A learned sink gives the head an explicit way to
attend to nothing. That makes it part of the model: gpt-oss ships one sink logit per
head in every attention layer of its checkpoint, and an engine that ignores them, or
renormalizes the weights "to be safe", or scales the sinks like scores, still runs
and produces fluent text from a different model. Fused attention kernels have to
know about the extra column too, since it cannot be added after the softmax: MLX's
`scaled_dot_product_attention` takes a `sinks` argument for it.
