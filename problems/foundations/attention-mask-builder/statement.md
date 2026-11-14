Build the attention mask for a batch of padded sequences: one tensor, added to the
attention scores, that hides the future from every token and hides padding from
everything. The difficulty is in the corners: which axis the padding mask belongs on,
where the padding is, and what happens to a row in which everything is hidden.

## Specification

```python
def build_mask(lengths, T, padding_side): ...
```

A batch holds `B` sequences padded to a common length `T`. Row `b` has `lengths[b]`
real tokens; the remaining positions are padding. Position `j` of row `b` is **real**
when

- `padding_side == "right"`: $j < \text{lengths}[b]$ (real tokens first, padding after);
- `padding_side == "left"`: $j \ge T - \text{lengths}[b]$ (padding first, real tokens
  last).

In attention, position `i` (the **query**) looks at position `j` (the **key**). Query
`i` of row `b` may attend to key `j` when either of these holds:

1. $j \le i$ and key `j` of row `b` is real: the key is not in the future, and it is
   not padding.
2. $j = i$: every position may attend to itself, even a padding position.

Rule 2 exists for padding queries. With left padding, every key at or before a padding
query is padding, so rule 1 alone would hide the entire row, and a softmax over a row
that is entirely $-\infty$ is `0 / 0`. For real queries rule 2 adds nothing.

The function returns the **additive** mask `M` of shape `(B, 1, T, T)`:

$$
M[b, 0, i, j] = \begin{cases}
0 & \text{if query } i \text{ of row } b \text{ may attend to key } j, \\
-\infty & \text{otherwise.}
\end{cases}
$$

It is meant to be added to scores of shape `(B, H, T, T)` before the softmax over the
last axis; the axis of length 1 broadcasts over the `H` heads.

- `lengths`: an integer tensor of shape `(B,)`.
- `T`: a positive Python int.
- `padding_side`: the string `"left"` or `"right"`. No other value is passed.
- Returns a float32 tensor of shape `(B, 1, T, T)` whose entries are exactly `0.0` or
  `-inf`. A large finite number such as `-1e9` is not accepted in place of `-inf`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `lengths` | `(B,)` | integer |
| result | `(B, 1, T, T)` | float32 |

## Example

Two rows padded to `T = 4`: the first is full, the second has two real tokens.
Writing `-` for `-inf`, with queries as rows and keys as columns:

```python
build_mask(lengths=[4, 2], T=4, padding_side="right")

# row 0 (no padding)      row 1 (real: 0, 1; padding: 2, 3)
# [[0, -, -, -],          [[0, -, -, -],
#  [0, 0, -, -],           [0, 0, -, -],
#  [0, 0, 0, -],           [0, 0, 0, -],     <- padding query 2: keys 0, 1, and itself
#  [0, 0, 0, 0]]           [0, 0, -, 0]]     <- padding query 3: key 2 is padding
```

```python
build_mask(lengths=[4, 2], T=4, padding_side="left")

# row 0 (no padding)      row 1 (padding: 0, 1; real: 2, 3)
# [[0, -, -, -],          [[0, -, -, -],     <- padding query 0: itself only
#  [0, 0, -, -],           [-, 0, -, -],     <- padding query 1: itself only
#  [0, 0, 0, -],           [-, -, 0, -],     <- real query 2: not the padding before it
#  [0, 0, 0, 0]]           [-, -, 0, 0]]
```

Row 0 is the plain causal mask in both cases. In row 1 with left padding, the columns
of the padding keys 0 and 1 are hidden from the real queries 2 and 3. Without rule 2,
the rows of queries 0 and 1 would contain no `0` at all.

The results have shape `(2, 1, 4, 4)`.

## Constraints

- The result is compared with the reference exactly: every entry is `0.0` or `-inf`.
- $1 \le B \le 16$, $1 \le T \le 16$, and $1 \le \text{lengths}[b] \le T$.
- No Python loops over batch items or positions are needed.

## Why this matters

Serving systems batch requests of different lengths into one tensor, and this mask is
the only thing that keeps the padding from changing the answers. Each mistake here has
a recognizable signature in production. Padding applied to queries instead of keys, or
on the wrong side, lets real tokens read padding embeddings, so the same prompt gives
different text depending on what it was batched with. A row that is entirely `-inf`
turns into NaN in the softmax; the NaN sits in a padding position at first, and then
spreads to real tokens through the next layer's attention. A finite stand-in for
`-inf` looks safe until two masks are added together or the model runs in half
precision. Batched generation uses left padding so that the last column is the last
real token of every row, which is exactly the case where whole rows of the mask are
hidden.
