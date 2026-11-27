Write three functions that make a tensor larger by repeating what is in it: key/value
heads shared by several query heads, batch rows copied once per beam, and one row of
positions for a whole batch. There are two orders in which copies can be laid out
along an axis, the frameworks disagree about which one the name `repeat` means, and
both orders return the same shape.

## Specification

```python
def repeat_kv_heads(k, n_rep): ...
def expand_for_beams(x, K): ...
def tile_positions(pos, B): ...
```

**`repeat_kv_heads(k, n_rep)`** takes `k` of shape `(B, G, T, D)`, the keys of `G`
groups, and returns `G * n_rep` heads, shape `(B, G * n_rep, T, D)`. Each group
serves `n_rep` consecutive heads:

$$
\text{out}[b,\ g \cdot n_{rep} + r] = k[b, g] \qquad \text{for } 0 \le r < n_{rep} .
$$

Head `h` of the result is group `h // n_rep`.

**`expand_for_beams(x, K)`** takes token ids `x` of shape `(B, T)` and returns `K`
consecutive copies of every row, shape `(B * K, T)`:

$$
\text{out}[b \cdot K + j] = x[b] \qquad \text{for } 0 \le j < K .
$$

The `K` beams of prompt `b` are rows `b * K` to `b * K + K - 1`.

**`tile_positions(pos, B)`** takes positions `pos` of shape `(T,)` and returns the
same row for every batch row, shape `(B, T)`:

$$
\text{out}[b, t] = \text{pos}[t] .
$$

- `k`: a float32 tensor; `repeat_kv_heads` returns float32.
- `x`, `pos`: integer tensors of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX). `expand_for_beams` and `tile_positions` return
  integer tensors; any integer width is accepted.
- `n_rep`, `K`, `B`: Python ints, at least 1. With `n_rep = 1` or `K = 1` the result
  has the values of the input.
- Results are compared by value. A result that shares memory with its input (an
  expanded view) is accepted, and so is a copy.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k` | `(B, G, T, D)` | float32 |
| result of `repeat_kv_heads` | `(B, G * n_rep, T, D)` | float32 |
| `x` | `(B, T)` | integer |
| result of `expand_for_beams` | `(B * K, T)` | integer |
| `pos` | `(T,)` | integer |
| result of `tile_positions` | `(B, T)` | integer |

## Tools

Repeating `[a, b]` twice along an axis has two possible results:

- **each element in place**, `[a, a, b, b]`: `torch.repeat_interleave(x, n, dim=d)`,
  `jnp.repeat(x, n, axis=d)`, `mx.repeat(x, n, axis=d)`;
- **the whole axis again**, `[a, b, a, b]`: `torch.tile(x, reps)` and
  `x.repeat(*reps)` in PyTorch, `jnp.tile(x, reps)`, `mx.tile(x, reps)`, with one
  count per axis.

> [!WARNING]
> `repeat` means the second in PyTorch and the first in JAX, MLX, and NumPy. Code
> ported between them keeps its shape and changes its order.

`x.expand(sizes)` in PyTorch and `broadcast_to(x, shape)` in all three repeat an axis
of size 1, or add leading axes. In PyTorch the result is a view that reads the same
memory many times, and nothing is copied. In MLX an evaluated `mx.broadcast_to`
added no memory either. In JAX `jnp.broadcast_to` took as long as `jnp.tile` when
measured, so there it saves nothing. That is the broadcasting of Broadcasting on Purpose made explicit, so it
needs the axis to exist with size 1 first (`None`). Merging the new axis into a
neighbor with `reshape`, as in Reshape, Transpose, and Memory Order, turns it into
one of the two orders above, and which one depends on the side of the neighbor the
new axis was put on.

## Example

```python
k = [[[[0.0, 1.0]], [[100.0, 101.0]]]]  # shape (1, 2, 1, 2): groups 0 and 1
repeat_kv_heads(k, n_rep=2)
# [[[[  0.0,   1.0]],               # head 0: group 0
#   [[  0.0,   1.0]],               # head 1: group 0
#   [[100.0, 101.0]],               # head 2: group 1
#   [[100.0, 101.0]]]]              # head 3: group 1; shape (1, 4, 1, 2)

x = [[1, 2, 3], [4, 5, 6]]  # shape (2, 3)
expand_for_beams(x, K=2)
# [[1, 2, 3],
#  [1, 2, 3],
#  [4, 5, 6],
#  [4, 5, 6]]                       # shape (4, 3)

pos = [0, 1, 2]  # shape (3,)
tile_positions(pos, B=2)
# [[0, 1, 2],
#  [0, 1, 2]]                       # shape (2, 3)
```

The other order would give heads `0, 1, 0, 1` and rows `[1, 2, 3], [4, 5, 6],
[1, 2, 3], [4, 5, 6]`, in the same shapes.

## Constraints

- Results are compared with the reference exactly: values are copied.
- `B`, `G`, `T`, and `D` are between 1 and 8, and `n_rep` and `K` between 1 and 4.
  Any of them can be 1, and any two can be equal.
- Token ids and positions lie in $[0, 1000)$.
- No Python loops are needed.

## Why this matters

Grouped-Query Attention repeats each key/value head for the query heads that share
it, in the order of `repeat_kv_heads`; tiled, most query heads attend with another
group's keys and the output still has the right shape. A
State-Space Step and Its Attention Form repeats its group parameters across heads the
same way. Sparse Expert Dispatch builds the token index of every (token, expert) pair
with `arange(n).repeat_interleave(k)`. Heavy-Hitter KV Eviction starts from
`arange(S)` expanded to `(B, H, S)`, which is `tile_positions` with one more axis.
A batched beam search copies every prompt `K` times before its first step, and its
bookkeeping (row `i` belongs to prompt `i // K`) assumes this order.
