## The idea

Repeating along an axis has two layouts, and they differ in which index moves
fastest. For two groups and three copies:

| | source of entry 0, 1, 2, 3, 4, 5 | entry `i` reads |
|---|---|---|
| each element in place (interleave) | `0, 0, 0, 1, 1, 1` | `i // n` |
| the whole axis again (tile) | `0, 1, 0, 1, 0, 1` | `i % G` |

The specification of `repeat_kv_heads` says head `g * n_rep + r` is group `g`, so
head `h` reads `h // n_rep`: the first layout. `expand_for_beams` says the same
about rows. `tile_positions` repeats a whole row, and since it does so along a new
axis there is only one layout to choose.

Both layouts come out of one construction, which is worth knowing because in PyTorch
it is also the form that copies nothing. Add an axis of size 1 next to the axis to repeat, expand
it, and merge the two with `reshape`. `reshape` reads in row-major order, last axis
fastest, so the new axis moves fastest when it is on the right:

```python
g = torch.arange(2)
g[:, None].expand(2, 3).reshape(6)  # [0, 0, 0, 1, 1, 1]: new axis after, interleave
g[None, :].expand(3, 2).reshape(6)  # [0, 1, 0, 1, 0, 1]: new axis before, tile
```

## Reference solution (PyTorch)

```python
def repeat_kv_heads(k, n_rep):
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x, K):
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos, B):
    return pos[None, :].expand(B, -1)
```

In JAX and MLX the first two are `repeat(k, n_rep, axis=1)` and
`repeat(x, K, axis=0)`, and the third is `broadcast_to(pos, (B, T))`.

The expand form of the first function:

```python
B, G, T, D = k.shape
k[:, :, None].expand(B, G, n_rep, T, D).reshape(B, G * n_rep, T, D)
```

## Expand is a view

`expand` and `repeat` return equal values, and the tests cannot tell them apart.
What differs is memory and what a write does. Measured on PyTorch 2.15, from a
`(1, 4096)` float32 tensor to `(4096, 4096)`:

| | storage | time |
|---|---|---|
| `x.expand(4096, 4096)` | 16 KiB, the storage of `x` | under 1 microsecond |
| `x.repeat(4096, 1)` | 64 MiB | about 0.6 ms |

The expanded tensor has a stride of 0 along the expanded axis: all 4096 rows are the
same 16 KiB. Three consequences, each run:

- **A write goes to every copy.** After `e = torch.zeros(1, 3).expand(4, 3)`,
  `e[0, 0] = 7.0` sets column 0 of all four rows, and of the tensor `e` was expanded
  from. `e += 1` and `e.add_(1)` raise `RuntimeError` ("more than one element of the
  written-to tensor refers to a single memory location"). Call `.clone()` before
  writing.
- **The view does not survive a merge.** `e.reshape(-1)` allocated the full 64 MiB,
  and `e.view(-1)` raises. So the expand form of `repeat_kv_heads` above copies at
  its `reshape`, and saves nothing over `repeat_interleave`. The saving is real
  when the expanded axis stays an axis: attention can keep keys as
  `(B, G, 1, T, D)` and let broadcasting against `(B, G, n_rep, T, D)` queries do
  the repeating.
- **Any arithmetic materializes it.** `e + 0` is an ordinary 64 MiB tensor.

In MLX an evaluated `mx.broadcast_to` of the same size added no active memory, and
neither did `mx.repeat` or `mx.tile` along an axis of size 1; repeating an axis
of size 4096 twice allocated the full result. A write into the broadcast array
(`e[0, 0] = 7.0`) changed one entry and left the source unchanged, so there is no
aliasing to guard against. JAX arrays are immutable, and `jnp.broadcast_to` took as
long as `jnp.tile` (1.1 ms each for this size, where `jnp.repeat` took 1.7 ms), so
it saves nothing there when called eagerly.

## Common mistakes

- **`repeat` for `repeat_interleave` in PyTorch.** `k.repeat(1, n_rep, 1, 1)` tiles:
  heads `0, 1, 0, 1`. The shape is right for every input, and the values are right
  when `G == 1` or `n_rep == 1`.
- **The new axis on the wrong side.** `k[:, None].expand(B, n_rep, G, T, D)` merged
  with `reshape` is the tile again. The new axis goes after the axis being repeated.
- **The wrong axis.** `repeat_interleave(n_rep, dim=2)` repeats positions and
  returns `(B, G, T * n_rep, D)`.
- **Tiling the batch for beams.** `x.repeat(K, 1)` gives rows `0, 1, 0, 1`. A batch
  of one cannot show the difference, and every later `beam // K` reads the wrong
  prompt.
- **No axis.** `x.repeat_interleave(K)` flattens first and returns `(B * T * K,)`.
  `jnp.repeat` and `mx.repeat` do the same without `axis`.
- **Positions repeated, then reshaped.** `pos.repeat_interleave(B)` is the `(T, B)`
  layout flattened; reshaping it to `(B, T)` gives rows like `0, 0, 1`. A reshape
  cannot fix an order.
- **The new axis after the positions.** `pos[:, None].expand(-1, B)` is `(T, B)`,
  the transpose, and has the right shape when `B == T`.
- **A bare `squeeze()` at the end.** On `tile_positions` it removes the batch axis of
  a batch of one and the position axis when `T == 1`. On `expand_for_beams` it
  returns `(B * K,)` for a single position and `(T,)` when `B * K == 1`.

## Complexity

Each function writes one value per entry of its result, or none for the expanded
view: $O(B\,G\,n_{rep}\,T\,D)$, $O(B\,K\,T)$, and $O(B\,T)$ or $O(1)$.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| each element in place | `x.repeat_interleave(n, dim=d)` | `jnp.repeat(x, n, axis=d)` | `mx.repeat(x, n, axis=d)` |
| the whole axis again | `x.repeat(*reps)`, `torch.tile(x, reps)` | `jnp.tile(x, reps)` | `mx.tile(x, reps)` |
| what `repeat` means | tile | each element | each element |
| no axis for the in-place repeat | flattens first | flattens first | flattens first |
| fewer counts than axes | `x.repeat`: `RuntimeError`. `torch.tile`: applies to the last axes | last axes | last axes |
| more counts than axes | new leading axes | new leading axes | new leading axes |
| a different count per element | a tensor of counts | an array of counts | `TypeError` |
| size 1 to size n by broadcasting | `x.expand(sizes)`, `torch.broadcast_to(x, shape)` | `jnp.broadcast_to(x, shape)` | `mx.broadcast_to(x, shape)` |
| what that costs, `(1, 4096)` to `(4096, 4096)` | a view: 16 KiB, under 1 microsecond | 1.1 ms, as long as `jnp.tile` | no added memory once evaluated |
| `-1` for "keep this size" | yes, in both | `ValueError` | `ValueError` |
| expanding an axis that is not 1 | `RuntimeError` | `ValueError` | `ValueError` |
| `e[0, 0] = 7` on the expanded result | every copy and the source change | `TypeError`: immutable | one entry changes, the source does not |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The row that causes bugs is the third. `Tensor.repeat` in PyTorch is `tile`
everywhere else, and `repeat` in NumPy, JAX, and MLX is PyTorch's
`repeat_interleave`. Porting `np.repeat(x, n, axis=1)` to `x.repeat(1, n, 1, 1)`, or
the reverse, changes the order and nothing else. The other difference is aliasing:
only PyTorch returns a view that writes through to every copy, so only PyTorch code
needs `.clone()` before an in-place update of an expanded tensor.
