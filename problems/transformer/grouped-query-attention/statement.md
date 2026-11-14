Implement grouped-query attention (GQA): multi-head attention in which several query
heads share one head of keys and values. The arithmetic is ordinary scaled dot-product
attention; the whole problem is pairing each query head with the right key/value head.

## Specification

```python
def gqa(q, k, v, mask): ...
```

The queries come in $H$ heads and the keys and values in $H_{kv}$ heads, where $H$ is a
multiple of $H_{kv}$. Let $G = H / H_{kv}$ be the group size. The query heads are
grouped in order: heads $0, \dots, G-1$ share key/value head 0, heads
$G, \dots, 2G-1$ share key/value head 1, and so on. Query head $h$ uses key/value head

$$
g(h) = \lfloor h / G \rfloor .
$$

For batch item $b$, query head $h$, query position $t$, and key position $s$, with $d$
the width of a head:

$$
\text{score}_{b,h,t,s} = \frac{q_{b,h,t} \cdot k_{b,g(h),s}}{\sqrt{d}} .
$$

The attention weights are a softmax over the key positions that the mask allows, and
the output is the weighted sum of the values of the same key/value head:

$$
a_{b,h,t,s} = \begin{cases}
\dfrac{e^{\text{score}_{b,h,t,s}}}{\sum_{s' :\, \text{mask}[t, s']} e^{\text{score}_{b,h,t,s'}}} & \text{mask}[t, s], \\[2ex]
0 & \text{otherwise},
\end{cases}
\qquad
y_{b,h,t} = \sum_{s} a_{b,h,t,s}\; v_{b,g(h),s} .
$$

- `q`: float32, shape `(B, H, T, d)`. `T` query positions.
- `k`, `v`: float32, shape `(B, H_kv, S, d)`. `S` key positions; `S` need not equal `T`
  (with a KV cache, `T` is the number of new tokens and `S` the length of the cache).
- `mask`: bool, shape `(T, S)`. `mask[t, s]` is `True` where query position `t` may
  attend to key position `s`. It is shared by all batch items and heads. Every row
  contains at least one `True`.
- Returns `y`: float32, shape `(B, H, T, d)`. The output has one head per **query**
  head.

The two ends of the range are ordinary cases, not special ones: $H_{kv} = H$ is
standard multi-head attention, and $H_{kv} = 1$ (every query head shares the single
key/value head) is multi-query attention.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call
a built-in attention function or layer (`scaled_dot_product_attention`,
`dot_product_attention`, `MultiheadAttention`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, H, T, d)` | float32 |
| `k` | `(B, H_kv, S, d)`, `H_kv` divides `H` | float32 |
| `v` | `(B, H_kv, S, d)` | float32 |
| `mask` | `(T, S)` | bool |
| result | `(B, H, T, d)` | float32 |

## Example

One batch item, $H = 4$ query heads, $H_{kv} = 2$ key/value heads (so $G = 2$), one
query position, two key positions, $d = 2$, and nothing masked:

```python
q = [[[[2.0, 0.0]], [[0.0, 2.0]], [[2.0, 0.0]], [[0.0, 2.0]]]]  # shape (1, 4, 1, 2)
k = [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, -1.0]]]]  # shape (1, 2, 2, 2)
v = [[[[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]]]]  # shape (1, 2, 2, 2)
mask = [[True, True]]  # shape (1, 2)

gqa(q, k, v, mask)
# [[[[0.804430, 0.195570]],
#   [[0.195570, 0.804430]],
#   [[8.044297, 1.955703]],
#   [[9.441928, 0.558072]]]]
```

Query heads 0 and 1 use key/value head 0; query heads 2 and 3 use key/value head 1.

- Head 0: scores $[2, 0] / \sqrt{2} = [1.4142, 0]$, weights $[0.8044, 0.1956]$, output
  $0.8044 \cdot [1, 0] + 0.1956 \cdot [0, 1]$.
- Head 1: scores $[0, 1.4142]$ against the same keys, so the weights are reversed.
- Head 2 has the same query as head 0 but reads key/value head 1: the keys $[1, 1]$
  and $[0, -1]$ give the scores $[1.4142, 0]$ again, and the values are ten times
  larger, so the output is $[8.0443, 1.9557]$.
- Head 3 has the same query as head 1 and also reads key/value head 1: scores
  $[1.4142, -1.4142]$, weights $[0.9442, 0.0558]$.

Heads 1 and 2 are the ones that tell the grouping apart: pair them with the other
key/value head and their outputs change.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le B \le 8$; $1 \le H \le 16$ and $H_{kv}$ divides $H$; $1 \le T, S \le 16$;
  $1 \le d \le 16$.
- Scores satisfy $|\text{score}| \le 100$.
- No Python loops over batch items, heads, or positions are needed.

## Why this matters

During generation the keys and values of every past token are kept in the KV cache,
for every layer and every head, and at long context that cache, not the weights, is
what fills the memory of a serving GPU. GQA (Ainslie et al., 2023) shrinks it by the
factor $G$ at almost no cost in quality, which is why nearly every current open-weight
model uses it: Llama 3 8B has 32 query heads and 8 key/value heads, so its cache is a
quarter of the size a 32-head cache would be. The price is one more convention that a
checkpoint does not spell out. Expanding the heads in the order `0, 1, 0, 1` instead of
`0, 0, 1, 1` produces tensors of the same shape, raises no error, and connects three
quarters of the query heads to keys they were never trained with.
