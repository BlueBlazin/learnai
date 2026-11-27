Implement `last_token_logits` to extract each row's logits at its **last real token** from a padded batch. This vector is the distribution of scores that a language model uses to sample the next token.

![This problem is the step after the final linear layer: taking each row's last real token.](assets/map.svg)

## Specification

```python
def last_token_logits(logits, lengths, padding_side): ...
```

A batch holds `B` sequences padded to a common length `T`. Row `b` has `lengths[b]` real tokens, and the model has produced `logits` containing a vector of `V` vocabulary scores for every position (including padding positions). The `padding_side` argument is either `"left"` or `"right"` (no other value is passed).

The index $p_b$ of the last real token in row `b` is:

$$
p_b = \begin{cases}
\text{lengths}[b] - 1 & \text{if padding\_side is "right" (real tokens at } 0, \dots, \text{lengths}[b] - 1\text{)}, \\
T - 1 & \text{if padding\_side is "left" (real tokens at } T - \text{lengths}[b], \dots, T - 1\text{)}.
\end{cases}
$$

Return a tensor `out` of shape `(B, V)` where each row `b` and vocabulary entry `v` is copied directly from position $p_b$:

$$
\text{out}[b, v] = \text{logits}[b, p_b, v] .
$$

Entries in `logits` may be `-inf` (for example, banned tokens or outputs at padding positions). Selected `-inf` values must be copied unchanged, and `-inf` values at unselected positions must not affect the result.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, T, V)` | float32 |
| `lengths` | `(B,)` | integer |
| result | `(B, V)` | float32 |

## Example

Consider two rows with `T = 3` and `V = 2`. Each entry is set to `100 * b + 10 * t + v` so its value encodes its row `b`, position `t`, and vocabulary index `v`:

```python
logits = [[[  0.0,   1.0], [ 10.0,  11.0], [ 20.0,  21.0]],    # row 0, positions 0, 1, 2
          [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]]]    # row 1, positions 0, 1, 2
lengths = [3, 1]

last_token_logits(logits, lengths, "right")  # [[20.0, 21.0], [100.0, 101.0]]
last_token_logits(logits, lengths, "left")   # [[20.0, 21.0], [120.0, 121.0]]
```

Row 0 is full, so its last real token is at position 2 under both padding sides. Row 1 has only one real token. With right padding, that token sits at position 0 while positions 1 and 2 are padding, so the result is `[100.0, 101.0]`. Reading `logits[:, -1]` would return `[120.0, 121.0]`, which comes from a padding position. With left padding, the single real token sits at position 2 (the last column), so the result is `[120.0, 121.0]`.

## Constraints

- The result is compared with the reference exactly, because values are copied rather than computed.
- $1 \le B, T, V \le 64$, and $1 \le \text{lengths}[b] \le T$.
- No Python loops over batch items are needed.

## Why this matters

Every decoding loop runs a prompt through the model, takes the logits at the end of each row, and samples the next token. For a single prompt, `logits[:, -1]` is correct, so most tutorials use it. On a right-padded batch, however, `logits[:, -1]` reads a padding position for every sequence shorter than the longest one. The longest prompt generates normally while shorter prompts produce plausible-looking nonsense, making the bug depend on which requests happen to share a batch.

Batched generation avoids this by padding on the left so the last real token of every row stays in the final column as new tokens are appended. Scoring and embedding pipelines usually pad on the right, so they must gather each row's last real token explicitly.
