With a KV cache, a model call sees only the new tokens, yet each of them must be
rotated by RoPE for its position in the **whole** sequence. Implement the two pieces of
bookkeeping that make this work in a batch: compute the absolute positions of the new
tokens from each row's cache length, and rotate the new queries and keys, and nothing
else, for those positions.

![This problem is RoPE positions, which rotates new tokens for their place in the whole sequence.](assets/map.svg)

## Specification

```python
def rope_positions(cache_lengths, n_new): ...  # returns positions
def rotate_new(q, k, positions, theta): ...  # returns (q_rot, k_rot)
```

**`rope_positions`.** Row `b` of the batch has `cache_lengths[b]` tokens in its cache,
at positions `0` to `cache_lengths[b] - 1`. The call appends `n_new` tokens to every
row. Return the absolute positions of those tokens:

$$
\text{positions}[b, i] = \text{cache\_lengths}[b] + i, \qquad 0 \le i < n_{\text{new}} .
$$

- `cache_lengths`: integer tensor, shape `(B,)`, values $\ge 0$. Rows may have
  different lengths, in any order; `0` means that nothing is cached for that row.
- `n_new`: a Python int, $\ge 1$.
- Returns an integer tensor of shape `(B, n_new)`.

**`rotate_new`.** Apply RoPE to the queries and keys of the new tokens, row `b` and
token `t` at position `positions[b, t]`. RoPE is the function of the
`rope` problem: the vector is split into
adjacent pairs $(x_{2i}, x_{2i+1})$, and pair $i$ is rotated counterclockwise by the
angle

$$
p \cdot \text{theta}^{-2i/D}, \qquad i = 0, \dots, D/2 - 1,
$$

where $p$ is the token's position.

- `q`, `k`: float32, shape `(B, H, T, D)`, `D` even: the queries and keys of the `T`
  new tokens, not yet rotated.
- `positions`: integer tensor, shape `(B, T)`: one row of positions per batch row,
  shared by all heads.
- `theta`: a Python float greater than 1, the RoPE base (the `base` argument of the
  `rope` problem; the `config.json` of a Llama checkpoint calls it `rope_theta`, and
  the value in the paper that introduced RoPE is 10000).
- Returns a tuple `(q_rot, k_rot)` of float32 tensors of shape `(B, H, T, D)`.

The cache does not appear in either signature, and that is the rule this problem
teaches: **cached keys are stored rotated.** A key is rotated once, when its token is
new, for that token's absolute position, and is then appended to the cache as it is.
It is never rotated again, because the score of a rotated query against a rotated key
depends only on the distance between their positions, and that distance never changes.
Values are not rotated at all.

Both functions are pure: they return new tensors and keep no state between calls. The
only state is the cache length, which the caller passes in, so the same code is
natural in PyTorch, JAX, and MLX.

You may call [`ops.rope(x, positions, base)`](../../../python/src/learnai/kit/ops/__init__.py)
from the kit (`from learnai.kit import ops`), which accepts positions of shape
`(T,)` or `(B, T)`, or write the rotation yourself.

## Shapes

| name | shape | dtype |
|---|---|---|
| `cache_lengths` | `(B,)` | integer |
| result of `rope_positions` | `(B, n_new)` | integer |
| `q`, `k` | `(B, H, T, D)`, `D` even | float32 |
| `positions` | `(B, T)` | integer |
| `q_rot`, `k_rot` | `(B, H, T, D)` | float32 |

## Example

Two rows: row 0 has three tokens cached, row 1 has none. Two new tokens arrive.

```python
positions = rope_positions([3, 0], 2)
# [[3, 4],
#  [0, 1]]
```

Take one head, `D = 4`, `theta = 100` (so the two pairs turn by $1$ and $0.1$ radian
per position), and let every new query and key be `[1, 0, 1, 0]`:

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

In row 0 the first new token is at position 3: its first pair $(1, 0)$ turns by $3$
radians to $(\cos 3, \sin 3) = (-0.9900, 0.1411)$ and its second pair by $0.3$ to
$(0.9553, 0.2955)$. In row 1 the same vector is at position 0 and does not turn.

The rows hold different numbers, and yet the score of the second new query against
the first new key is $\cos 1 + \cos 0.1 = 1.5353$ in both: the two tokens are one
position apart in either row. Starting row 0 at position 0 instead of 3 would give the
same score **within this call**, and wrong scores against every key already in the
cache, which was rotated for positions 0 to 2.

![The second new query and the first new key of the example. At shift 0 they are row 1, at shift 3 row 0, with the same score.](interactive:rope-shift)

## Constraints

- $1 \le B, H \le 4$; $1 \le T,\ n_{\text{new}} \le 4$; $D$ even, $2 \le D \le 16$;
  $10 \le \text{theta} \le 10^4$; every position is below 2048.
- `rope_positions` is compared exactly.
- `rotate_new` is compared with the reference, and with RoPE applied in one pass to the
  whole sequence. When every position is below 64: relative tolerance `1e-4`, absolute
  tolerance `1e-4`. With larger positions: relative tolerance `1e-3`, absolute
  tolerance `5e-3`, because the angle is a float32 product whose rounding error grows
  with the position.
- No Python loops over rows, tokens, or pairs are needed.

## Why this matters

Position bookkeeping is where cached generation goes wrong without any error. A
prefill pass starts every row at position 0, so code that numbers tokens with
`arange(T)` passes every test that runs the whole prompt at once, and then gives every
generated token position 0. The model answers the prompt and drifts into nonsense a
few tokens later. Batching adds a second trap: rows of one batch are at different
lengths as soon as prompts differ or requests join a running batch, so a single
offset for the whole batch is right for one row only. And an engine that
"refreshes" the cached keys by rotating the concatenated tensor again turns every
old key a second time. Real engines carry a position per token for this reason:
llama.cpp's batch has a `pos` entry for each token, and Hugging Face models take
`position_ids` of shape `(B, T)`.
