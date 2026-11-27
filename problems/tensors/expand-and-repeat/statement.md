Implement three functions that enlarge a tensor by repeating its elements: duplicating key/value heads for shared query heads, copying batch rows for beam search, and broadcasting a single position vector across a batch. Copies along an axis can be laid out in two different orders that produce the exact same shape, and frameworks disagree on which order `repeat` refers to.

## Specification

```python
def repeat_kv_heads(k, n_rep): ...
def expand_for_beams(x, K): ...
def tile_positions(pos, B): ...
```

**`repeat_kv_heads(k, n_rep)`** repeats key/value head groups so that each group serves `n_rep` consecutive query heads. Here `B` is the batch size, `G` is the number of key/value groups, `T` is the sequence length, and `D` is the head dimension. The output has `G * n_rep` heads, where group `g` fills `n_rep` adjacent slots:

$$
\text{out}[b,\ g \cdot n_{\text{rep}} + r] = k[b, g] \qquad \text{for } 0 \le r < n_{\text{rep}} .
$$

Equivalently, output head `h` reads from group `h // n_rep`.

**`expand_for_beams(x, K)`** duplicates each row of token IDs `x` (of `B` prompts and length `T`) `K` times consecutively along the batch axis:

$$
\text{out}[b \cdot K + j] = x[b] \qquad \text{for } 0 \le j < K .
$$

The `K` beams for prompt `b` occupy rows `b * K` through `b * K + K - 1`.

**`tile_positions(pos, B)`** repeats a 1D vector of `T` position indices across a new leading batch axis of size `B`:

$$
\text{out}[b, t] = \text{pos}[t] .
$$

Additional requirements:

- `x` and `pos` use your framework's default integer dtype (`int64` in PyTorch, `int32` in JAX and MLX). `expand_for_beams` and `tile_positions` may return any integer width.
- `n_rep`, `K`, and `B` are Python `int`s of at least `1`. When `n_rep == 1` or `K == 1`, the output values match the input.
- Results are compared by value. Both an expanded view (sharing memory with the input) and a freshly allocated copy are accepted.

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

Repeating `[a, b]` twice along an axis produces one of two layouts:

- **Interleaving each element in place** gives `[a, a, b, b]`. Use `torch.repeat_interleave(x, n, dim=d)`, `jnp.repeat(x, n, axis=d)`, or `mx.repeat(x, n, axis=d)`.
- **Tiling the whole axis** gives `[a, b, a, b]`. Use `torch.tile(x, reps)` or `x.repeat(*reps)` in PyTorch, `jnp.tile(x, reps)` in JAX, or `mx.tile(x, reps)` in MLX, supplying one repetition count per axis.

> [!WARNING]
> `repeat` tiles the whole axis in PyTorch, but interleaves each element in JAX, MLX, and NumPy. Porting code between them preserves the output shape while silently changing the element order.

To repeat a size-1 axis or add leading axes, use `x.expand(sizes)` in PyTorch or `broadcast_to(x, shape)` in any of the three frameworks. In PyTorch, `expand` returns a zero-copy view that reads the same memory repeatedly. In MLX, an evaluated `mx.broadcast_to` also adds no extra memory. In JAX, `jnp.broadcast_to` takes about the same time as `jnp.tile` when run eagerly.

To repeat an axis whose size is greater than `1` using `expand` or `broadcast_to`, insert a size-1 axis with `None`, expand it, and merge it into the adjacent axis with `reshape`. Whether the merged result interleaves or tiles depends on which side of the target axis you place the new size-1 axis.

## Example

```python
k = [[[[0.0, 1.0]], [[100.0, 101.0]]]]  # shape (1, 2, 1, 2), groups 0 and 1
repeat_kv_heads(k, n_rep=2)
# [[[[  0.0,   1.0]],               # head 0 from group 0
#   [[  0.0,   1.0]],               # head 1 from group 0
#   [[100.0, 101.0]],               # head 2 from group 1
#   [[100.0, 101.0]]]]              # head 3 from group 1, shape (1, 4, 1, 2)

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

Using the tiled order instead would produce heads `0, 1, 0, 1` and rows `[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]`, with the exact same shapes.

## Constraints

- Results are compared against the reference for exact equality because values are only copied.
- `B`, `G`, `T`, and `D` are between `1` and `8`, and `n_rep` and `K` are between `1` and `4`. Any dimension can be `1`, and any two dimensions can be equal.
- Token IDs and positions lie in $[0, 1000)$.
- No Python loops are needed.

## Why this matters

Grouped-query attention repeats each key/value head for the query heads that share it, using the interleaved order of `repeat_kv_heads`. If you tile instead, most query heads attend to another group's keys while the output still has the expected shape. State-space models repeat group parameters across heads the same way. Sparse expert dispatch builds token indices for `(token, expert)` pairs with `repeat_interleave`, and heavy-hitter KV eviction expands a 1D vector of sequence positions across both batch and head axes, just like `tile_positions` with an extra axis.

Batched beam search duplicates every prompt `K` times before decoding. All downstream bookkeeping relies on row `i` belonging to prompt `i // K`, which silently breaks if the rows are tiled instead of interleaved.
