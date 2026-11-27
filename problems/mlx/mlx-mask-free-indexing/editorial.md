## The idea

MLX constructs a computation graph before running any operations, so every array in the graph must have a shape known without inspecting values. Boolean-mask reads like `x[mask]` are unsupported because the output length equals the number of `True` entries in `mask`, which is only known after evaluation. Every replacement MLX provides has a statically known output shape:

| instead of | use | result shape |
|---|---|---|
| `x[mask] = v` on a copy | `mx.where(mask, v, x)` | the shape of `x` |
| `x[mask]` | keep the mask, or `mx.where` with a neutral value | the shape of `x` |
| "the top k" | `mx.argpartition(-x, kth=k - 1)[..., :k]`, `mx.topk(x, k)` | `(..., k)` |
| `x.gather(1, idx)` | `mx.take_along_axis(x, idx, axis)` | the shape of `idx` |
| `x.scatter(1, idx, v)` | `mx.put_along_axis(x, idx, v, axis)` | the shape of `x` |
| `x.scatter_add(1, idx, v)` | `x.at[rows, idx].add(v)` | the shape of `x` |

**Top-k.** Partitioning `-logits` around index `k - 1` places the indices of the `k` largest logits in the first `k` slots and the remaining `V - k` indices after them. Scattering `-inf` into those `V - k` positions completes the filter. Existing `-inf` entries need no special handling because negating them produces `+inf`, which partitions to the end and gets overwritten with `-inf` anyway.

**Penalty.** Gather the logits at `ids`, subtract `value`, and scatter the updated values back to the same positions. Although the MLX documentation notes that concurrent writes to the same location are nondeterministic, every duplicate index here writes the exact same value (`logits[b, id] - value`). The result is therefore deterministic, and each unique token in `ids` is penalized exactly once.

Neither function reads array values, so neither forces evaluation. Checking `k >= V` inspects only the array's shape, which is already known for a lazy array.

## Reference solution (MLX)

```python
def top_k_filter(logits, k):
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits
    order = mx.argpartition(-logits, kth=k - 1, axis=-1)
    minus_inf = mx.array(-math.inf, dtype=logits.dtype)
    return mx.put_along_axis(logits, order[..., k:], minus_inf, axis=-1)


def scatter_penalty(logits, ids, value):
    picked = mx.take_along_axis(logits, ids, axis=-1)
    return mx.put_along_axis(logits, ids, picked - value, axis=-1)
```

Because finite logits in each row are distinct here, a threshold-based top-k using `mx.topk` and `mx.where` also works. Compute the smallest of the `k` largest values and mask everything below it:

```python
kth = mx.min(mx.topk(logits, k, axis=-1), axis=-1, keepdims=True)
return mx.where(logits >= kth, logits, -math.inf)
```

When ties occur at the threshold, the threshold approach keeps more than `k` entries, which is why `top-k-filtering` ranks by index. `mlx_lm.sample_utils.apply_top_k` uses the `argpartition` and `put_along_axis` form above.

## Common mistakes

- **Round-tripping through NumPy.** Calling `np.array(logits)` evaluates the graph, copies logits to the CPU, and requires uploading the filtered result back to the GPU. The numbers match, but on a 151,936-token vocabulary the step slows from 2.3 ms to 5.2 ms.
- **Reading through a boolean mask.** `out[keep] = logits[keep]` fails on the right-hand side with `ValueError: boolean indices are not yet supported`. Boolean masks work for assignment only.
- **Accumulating updates for the penalty.** `logits.at[rows, ids].add(-value)` is MLX's `scatter_add`, so a token listed four times loses `4 * value`. That belongs in a frequency penalty, not a presence penalty. By contrast, `x[rows, ids] -= value` does not accumulate because it performs one gather, one subtraction, and one scatter.
- **Mutating the input array.** `logits[rows, ids] = ...` produces the right numbers while corrupting the caller's array, which other sampler stages or log-probability calculations may still need.
- **Treating `argpartition` as `argsort`.** Only position `kth` is guaranteed to hold its sorted element. On MLX 0.32.4's GPU backend, `argpartition` happens to return fully sorted indices regardless of `kth`, so `kth=0` passes every GPU test. On the CPU device, however, it keeps the wrong tokens.
- **Assuming `mx.topk` returns sorted values like `torch.topk`.** PyTorch sorts `topk` outputs in descending order, so `values[..., -1:]` is the $k$-th largest element. MLX makes no ordering guarantee — in practice the last element is often the *largest*, which causes the threshold to keep only one token.
- **Off-by-one error in `kth`.** Partitioning around `k` and masking from `k + 1` keeps `k + 1` logits instead of `k`.
- **Missing the $k \ge V$ guard.** `kth = k - 1` must be a valid index, so `mx.argpartition` raises an error whenever $k \ge V$.
- **Evaluating before returning.** Calling `mx.eval(out)` inside a processor forces an extra GPU sync per processor per token. Evaluation timing belongs to the caller (see `mlx-lazy-eval-placement`).

## Complexity

`argpartition` runs in $O(V)$ average time per row (selection rather than a full sort), and `put_along_axis` writes $V - k$ entries. `scatter_penalty` runs in $O(N)$ per row. Both functions allocate one new `(B, V)` array.

## In the wild

`mlx_lm.sample_utils` uses these same patterns across its samplers. `apply_top_k` partitions and scatters `-inf`. `apply_top_p` sorts, takes a cumulative sum, gathers a threshold with `mx.take_along_axis`, and selects with `mx.where`. The repetition penalty gathers recent token logits, rescales them, and scatters them back. This shape-first style is also required by `mx.compile` later in this track, since any function whose output shape depends on array values cannot be traced once and replayed.
