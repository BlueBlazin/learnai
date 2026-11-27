With a KV cache, each forward pass receives only the new tokens, but RoPE must rotate
every token using its absolute position in the full sequence. Implement two pure
functions for batched inference. The first computes each new token's absolute position
from its row's cache length, and the second rotates only the new queries and keys for
those positions.

![This problem is RoPE positions, which rotates new tokens for their place in the whole sequence.](assets/map.svg)

## Specification

```python
def rope_positions(cache_lengths, n_new): ...  # returns positions
def rotate_new(q, k, positions, theta): ...  # returns (q_rot, k_rot)
```

**`rope_positions`.** Let `B` be the batch size. Row `b` currently holds
`cache_lengths[b]` cached tokens at positions `0` through `cache_lengths[b] - 1`.
When `n_new` new tokens arrive for every row, return their absolute positions:

$$
\text{positions}[b, i] = \text{cache\_lengths}[b] + i, \qquad 0 \le i < n_{\text{new}}.
$$

- `cache_lengths`: non-negative integers, one per row. Rows may have different lengths
  in any order, and `0` means the row's cache is empty.
- `n_new`: Python integer $\ge 1$.
- Returns `positions`, with one row of `n_new` absolute positions per batch item.

**`rotate_new`.** Given unrotated queries `q` and keys `k` of shape `(B, H, T, D)`
(batch size `B`, attention heads `H`, new tokens `T`, and even head dimension `D`),
apply RoPE to each token `(b, t)` at position $p = \text{positions}[b, t]$ across all
heads. As in the `rope` problem, split each vector of length `D` into adjacent pairs
$(x_{2i}, x_{2i+1})$ and rotate pair $i$ counterclockwise by the angle

$$
p \cdot \text{theta}^{-2i/D}, \qquad i = 0, \dots, D/2 - 1.
$$

- `q`, `k`: unrotated queries and keys for the `T` new tokens.
- `positions`: one sequence of positions per batch row, shared by all heads.
- `theta`: Python float $> 1$, the RoPE base. This is the `base` argument from the
  `rope` problem and `rope_theta` in a Llama `config.json` (10000 in the original
  RoPE paper).
- Returns `(q_rot, k_rot)`, the rotated queries and keys.

Neither function takes the KV cache as an argument because **cached keys are stored
already rotated**. Each key is rotated once for its absolute position when its token
arrives, then appended to the cache as is. Because the attention score between a
rotated query and a rotated key depends only on the distance between their positions,
cached keys are never rotated again. Values are not rotated at all.

Both functions are pure and keep no state between calls. The caller tracks each row's
cache length and passes it in, so the same design works directly in PyTorch, JAX, and
MLX.

You may call [`ops.rope(x, positions, base)`](../../../python/src/learnai/kit/ops/__init__.py)
from the kit (`from learnai.kit import ops`), which accepts `positions` of shape
`(T,)` or `(B, T)`, or implement the rotation yourself.

## Shapes

| name | shape | dtype |
|---|---|---|
| `cache_lengths` | `(B,)` | integer |
| result of `rope_positions` | `(B, n_new)` | integer |
| `q`, `k` | `(B, H, T, D)`, `D` even | float32 |
| `positions` | `(B, T)` | integer |
| `q_rot`, `k_rot` | `(B, H, T, D)` | float32 |

## Example

Consider a batch of two rows where row 0 has 3 cached tokens, row 1 has 0 cached
tokens, and 2 new tokens arrive.

```python
positions = rope_positions([3, 0], 2)
# [[3, 4],
#  [0, 1]]
```

Now take `H = 1`, `D = 4`, and `theta = 100`, so the two pairs rotate by $1$ and
$0.1$ radians per position. Let every new query and key be `[1, 0, 1, 0]`:

```python
x = [
    [[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
    [[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
]  # shape (2, 1, 2, 4)

q_rot, k_rot = rotate_new(x, x, positions, 100.0)
# q_rot = k_rot =
# [[[[-0.989992,  0.141120,  0.955337,  0.295520],          row 0, position 3
#    [-0.653644, -0.756802,  0.921061,  0.389418]]],        row 0, position 4
#  [[[ 1.000000,  0.000000,  1.000000,  0.000000],          row 1, position 0
#    [ 0.540302,  0.841471,  0.995004,  0.099833]]]]        row 1, position 1
```

In row 0, the first new token sits at position 3. Its first pair $(1, 0)$ rotates by
$3$ radians to $(\cos 3, \sin 3) = (-0.9900, 0.1411)$, and its second pair rotates by
$0.3$ radians to $(0.9553, 0.2955)$. In row 1, the same vector sits at position 0 and
does not rotate.

Even though the two rows contain different numbers, the dot product between the second
new query and the first new key is $\cos 1 + \cos 0.1 = 1.5353$ in both rows because
the two tokens are one position apart. If row 0 erroneously started at position 0
instead of 3, its scores **within this call** would still look right, but its scores
against the cached keys at positions 0 to 2 would be wrong.

![The second new query and the first new key of the example. At shift 0 they are row 1, at shift 3 row 0, with the same score.](interactive:rope-shift)

## Constraints

- $1 \le B, H \le 4$, $1 \le T, n_{\text{new}} \le 4$, $D$ is even with
  $2 \le D \le 16$, $10 \le \text{theta} \le 10^4$, and every position is below 2048.
- `rope_positions` is checked for exact equality.
- `rotate_new` is compared against the reference and against RoPE applied in a single
  pass to the whole sequence. When every position is below 64, the tolerances are
  `1e-4` relative and `1e-4` absolute. For larger positions, the tolerances are `1e-3`
  relative and `5e-3` absolute because float32 rounding error in the angle product
  grows with position.
- No Python loops over rows, tokens, or pairs are needed.

## Why this matters

Position bugs are a common source of silent failure in cached decoding. Because
prefill starts every row at position 0, code that numbers tokens with `arange(T)`
passes every full-prompt test and then assigns position 0 to every generated token.
The model responds coherently at first and drifts into nonsense a few tokens later.

Batching introduces a second pitfall. Rows in a batch have different lengths whenever
prompts differ or new requests join a running batch, so a single scalar offset is
valid for only one row. Likewise, concatenating keys before applying RoPE rotates
every cached key a second time. Production engines therefore track explicit positions
per token. For example, `llama.cpp` stores a `pos` field for every token in a batch,
and Hugging Face models accept `position_ids` of shape `(B, T)`.
