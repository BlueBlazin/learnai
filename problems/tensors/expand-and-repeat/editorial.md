## The idea

Repeating along an axis has two possible layouts, distinguished by which index varies fastest. For two groups repeated three times:

| | source of entry 0, 1, 2, 3, 4, 5 | entry `i` reads |
|---|---|---|
| each element in place (interleave) | `0, 0, 0, 1, 1, 1` | `i // n` |
| the whole axis again (tile) | `0, 1, 0, 1, 0, 1` | `i % G` |

In `repeat_kv_heads`, head `g * n_rep + r` comes from group `g`, so head `h` reads `h // n_rep`. That is the interleaved layout. `expand_for_beams` uses the same interleaved layout along the batch axis. `tile_positions` repeats a whole row along a brand-new leading axis, so there is only one layout.

Both layouts come from a single construction: insert a size-1 axis next to the axis you want to repeat, expand it, and merge the two axes with `reshape`. Because `reshape` reads in row-major order (last axis fastest), placing the new axis on the right interleaves, and placing it on the left tiles:

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

In JAX and MLX, the first two functions are `repeat(k, n_rep, axis=1)` and `repeat(x, K, axis=0)`, and the third is `broadcast_to(pos, (B, T))`.

You can also write `repeat_kv_heads` using `expand` and `reshape`:

```python
B, G, T, D = k.shape
k[:, :, None].expand(B, G, n_rep, T, D).reshape(B, G * n_rep, T, D)
```

## Expand is a view

`expand` and `repeat` return equal values, so value-based tests cannot tell them apart. They differ in memory usage and write behavior. Measured on PyTorch 2.15 when expanding a `(1, 4096)` float32 tensor to `(4096, 4096)`:

| | storage | time |
|---|---|---|
| `x.expand(4096, 4096)` | 16 KiB (the storage of `x`) | under 1 microsecond |
| `x.repeat(4096, 1)` | 64 MiB | about 0.6 ms |

The expanded tensor sets a stride of `0` along the expanded axis, so all 4096 rows point to the same 16 KiB of memory. This has three practical consequences in PyTorch:

- **In-place writes affect every copy.** After `e = torch.zeros(1, 3).expand(4, 3)`, assigning `e[0, 0] = 7.0` changes column `0` in all four rows as well as in the original tensor. Operations like `e += 1` and `e.add_(1)` raise `RuntimeError` ("more than one element of the written-to tensor refers to a single memory location"). Call `.clone()` before mutating an expanded tensor.
- **Merging axes forces a copy.** Calling `e.reshape(-1)` allocates the full 64 MiB, and `e.view(-1)` raises an error. Because of this, the `expand` + `reshape` version of `repeat_kv_heads` still copies at `reshape` and saves no memory over `repeat_interleave`. The memory saving only holds if you keep the expanded axis separate, such as keeping keys as `(B, G, 1, T, D)` and letting broadcasting against `(B, G, n_rep, T, D)` queries handle the repetition.
- **Arithmetic materializes the tensor.** Computing `e + 0` allocates an ordinary 64 MiB tensor.

In MLX, an evaluated `mx.broadcast_to` of the same size adds no active memory, and neither does `mx.repeat` or `mx.tile` along a size-1 axis. Repeating an axis of size 4096 twice allocates the full result. Writing `e[0, 0] = 7.0` into a broadcast MLX array updates only that entry and leaves the source untouched, so there is no aliasing hazard. JAX arrays are immutable, and `jnp.broadcast_to` took as long as `jnp.tile` when called eagerly (1.1 ms each at this size, compared to 1.7 ms for `jnp.repeat`), so it saves nothing outside JIT fusion.

## Common mistakes

- **Using `repeat` instead of `repeat_interleave` in PyTorch.** `k.repeat(1, n_rep, 1, 1)` tiles the heads into `0, 1, 0, 1`. The shape is always right, and the values even pass when `G == 1` or `n_rep == 1`.
- **Inserting the new axis on the wrong side.** `k[:, None].expand(B, n_rep, G, T, D)` followed by `reshape` tiles instead of interleaving. Place the new size-1 axis immediately after the axis being repeated.
- **Repeating along the wrong axis.** `repeat_interleave(n_rep, dim=2)` repeats sequence positions and returns `(B, G, T * n_rep, D)`.
- **Tiling the batch for beams.** `x.repeat(K, 1)` produces rows `0, 1, 0, 1`. A batch of size `1` hides the bug, and any downstream `beam // K` lookup reads the wrong prompt.
- **Omitting the axis argument.** Calling `x.repeat_interleave(K)` without `dim` flattens the tensor first and returns `(B * T * K,)`. `jnp.repeat` and `mx.repeat` do the same when `axis` is omitted.
- **Repeating positions and reshaping.** `pos.repeat_interleave(B)` flattens the `(T, B)` layout, so reshaping it to `(B, T)` produces corrupted rows like `[0, 0, 1]`. A `reshape` cannot fix the wrong element order.
- **Adding the new axis after positions.** `pos[:, None].expand(-1, B)` produces `(T, B)` (the transpose), which still has the expected shape whenever `B == T`.
- **Calling bare `squeeze()` at the end.** On `tile_positions`, `.squeeze()` drops the batch axis when `B == 1` or the position axis when `T == 1`. On `expand_for_beams`, it returns `(B * K,)` when `T == 1` and `(T,)` when `B * K == 1`.

## Complexity

Each function writes one value per output element, except for an expanded view that does $O(1)$ work. The costs are $O(B\,G\,n_{\text{rep}}\,T\,D)$ for `repeat_kv_heads`, $O(B\,K\,T)$ for `expand_for_beams`, and $O(B\,T)$ (or $O(1)$ as a view) for `tile_positions`.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| each element in place | `x.repeat_interleave(n, dim=d)` | `jnp.repeat(x, n, axis=d)` | `mx.repeat(x, n, axis=d)` |
| the whole axis again | `x.repeat(*reps)`, `torch.tile(x, reps)` | `jnp.tile(x, reps)` | `mx.tile(x, reps)` |
| what `repeat` means | tile | each element | each element |
| no axis for the in-place repeat | flattens first | flattens first | flattens first |
| fewer counts than axes | `x.repeat` raises `RuntimeError`; `torch.tile` applies to last axes | last axes | last axes |
| more counts than axes | new leading axes | new leading axes | new leading axes |
| a different count per element | a tensor of counts | an array of counts | `TypeError` |
| size 1 to size n by broadcasting | `x.expand(sizes)`, `torch.broadcast_to(x, shape)` | `jnp.broadcast_to(x, shape)` | `mx.broadcast_to(x, shape)` |
| cost for `(1, 4096)` to `(4096, 4096)` | view: 16 KiB, under 1 microsecond | 1.1 ms (same as `jnp.tile`) | no added memory once evaluated |
| `-1` for "keep this size" | yes, in both | `ValueError` | `ValueError` |
| expanding an axis that is not 1 | `RuntimeError` | `ValueError` | `ValueError` |
| `e[0, 0] = 7` on the expanded result | every copy and the source change | `TypeError` (immutable) | one entry changes, source unchanged |

Every entry was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The most common source of bugs is the third row. `Tensor.repeat` in PyTorch behaves like `tile` everywhere else, whereas `repeat` in NumPy, JAX, and MLX matches PyTorch's `repeat_interleave`. Translating `np.repeat(x, n, axis=1)` to `x.repeat(1, n, 1, 1)` (or the reverse) silently flips the element order while keeping the shape identical. The other key difference is aliasing. Only PyTorch returns a view where in-place writes propagate to every copy, so only PyTorch requires `.clone()` before mutating an expanded tensor.
