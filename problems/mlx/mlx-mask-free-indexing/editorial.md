## The idea

MLX builds a graph before it computes anything, so every array in the graph must have
a shape that is known without looking at values. That is the reason behind the
missing idiom: `x[mask]` has as many elements as the mask has `True` entries, a number
that exists only after evaluation. Everything MLX offers instead has a shape fixed in
advance:

| instead of | use | result shape |
|---|---|---|
| `x[mask] = v` on a copy | `mx.where(mask, v, x)` | the shape of `x` |
| `x[mask]` | keep the mask, or `mx.where` with a neutral value | the shape of `x` |
| "the top k" | `mx.argpartition(-x, kth=k - 1)[..., :k]`, `mx.topk(x, k)` | `(..., k)` |
| `x.gather(1, idx)` | `mx.take_along_axis(x, idx, axis)` | the shape of `idx` |
| `x.scatter(1, idx, v)` | `mx.put_along_axis(x, idx, v, axis)` | the shape of `x` |
| `x.scatter_add(1, idx, v)` | `x.at[rows, idx].add(v)` | the shape of `x` |

**Top-k.** Partitioning the negated logits around position `k - 1` puts the indices of
the `k` largest logits first and the indices of the other `V - k` after them. Writing
`-inf` at those `V - k` positions is the whole filter. Rows with `-inf` entries need no
special case: negated, they are `+inf` and sort to the end, among the positions that
are overwritten with `-inf` anyway.

**Penalty.** Gather the logits at `ids`, subtract, scatter the results to the same
positions. The MLX documentation warns that "updates to the same location are
nondeterministic": with repeated indices, which write lands last is undefined. Here
all writes to one location carry the same value, so the result is deterministic and
every named token is penalized exactly once.

Neither function asks for a value, so neither evaluates: the comparison `k >= V` reads
a shape, which is known for a lazy array.

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

A threshold version of top-k is equally good here, because the finite logits of a row
are distinct: take the smallest of the `k` largest values as the threshold and select
with `mx.where`.

```python
kth = mx.min(mx.topk(logits, k, axis=-1), axis=-1, keepdims=True)
return mx.where(logits >= kth, logits, -math.inf)
```

With ties at the threshold it keeps more than `k` entries, which is why
`top-k-filtering` ranks by index; `mlx_lm.sample_utils.apply_top_k` uses the
`argpartition` and `put_along_axis` form above.

## Common mistakes

- **A round trip through NumPy.** `np.array(logits)` evaluates the graph, copies the
  logits out, and the result has to be uploaded again. The numbers are right and the
  step is cut in two; on a 151,936-token vocabulary on this machine the step takes 5.2
  ms instead of 2.3 ms.
- **Reading through a boolean mask.** `out[keep] = logits[keep]` fails on the right
  side: `ValueError: boolean indices are not yet supported`. Boolean masks work for
  assignment only.
- **Accumulating updates for the penalty.** `logits.at[rows, ids].add(-value)` is the
  MLX spelling of `scatter_add`: a token named four times loses four times the
  penalty. Use it when you want counts (a frequency penalty), not here. Note the
  asymmetry with the in-place form: `x[rows, ids] -= value` does *not* accumulate,
  because it is a gather, a subtraction, and one scatter.
- **Assigning into the argument.** `logits[rows, ids] = ...` gives the right numbers
  and changes the caller's array, which another sampler stage, or the code computing
  log-probabilities for the response, was going to read.
- **Treating `argpartition` as `argsort`.** Only the position `kth` is guaranteed.
  On this machine's GPU, MLX 0.32.4 returns the indices fully sorted whatever `kth`
  is, so `kth=0` passes every GPU test; on the CPU device the same code keeps the
  wrong tokens. Code that relies on an undefined order is correct until the backend
  changes.
- **Assuming `mx.topk` is sorted like `torch.topk`.** PyTorch returns the `k` values
  in descending order, so `values[..., -1:]` is the k-th largest. MLX documents no
  order; in practice the last element is the *largest*, and the "threshold" keeps one
  token.
- **Off by one in `kth`.** Partitioning around `k` and masking from `k + 1` keeps
  `k + 1` logits.
- **No guard for $k \ge V$.** `kth = k - 1` must be a valid position:
  `mx.argpartition` raises for `kth >= V`.
- **Evaluating the result before returning it.** `mx.eval(out)` inside a processor
  forces an evaluation per processor per token. Where to evaluate is the caller's
  decision (see `mlx-lazy-eval-placement`).

## Complexity

`argpartition` is $O(V)$ per row on average (a selection, not a sort) and the scatter
writes $V - k$ values; the penalty is $O(N)$ per row. Both allocate one new `(B, V)`
array.

## In the wild

`mlx_lm.sample_utils` builds its samplers this way: `apply_top_k` partitions and
scatters `-inf`, `apply_top_p` sorts, takes a cumulative sum, gathers a threshold
with `mx.take_along_axis`, and selects with `mx.where`, and the repetition penalty
gathers the logits of the recent tokens, rescales them, and assigns them back. The same shape-first thinking
is what `mx.compile` requires later in this track: a function whose output shapes
depend on values cannot be traced once and replayed.
