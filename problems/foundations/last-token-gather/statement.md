Given the logits of a padded batch, return for each row the logits at its **last real
token**: the vector that the next token is sampled from. It is one line of indexing,
and that line is the classic reason why batched generation gives garbage for the short
prompts in a batch.

## Specification

```python
def last_token_logits(logits, lengths, padding_side): ...
```

A batch holds `B` sequences padded to a common length `T`. Row `b` has `lengths[b]`
real tokens. A model has produced `logits` of shape `(B, T, V)`: one vector of `V`
scores for every position, padding positions included.

The position $p_b$ of the last real token of row `b` is

$$
p_b = \begin{cases}
\text{lengths}[b] - 1 & \text{if padding\_side is "right" (real tokens at } 0, \dots, \text{lengths}[b] - 1\text{)}, \\
T - 1 & \text{if padding\_side is "left" (real tokens at } T - \text{lengths}[b], \dots, T - 1\text{)}.
\end{cases}
$$

The function returns `out` of shape `(B, V)` with

$$
\text{out}[b, v] = \text{logits}[b, p_b, v] .
$$

- `logits`: a float32 tensor of shape `(B, T, V)`. Entries may be `-inf` (banned
  tokens, or whatever a model wrote at a padding position); they are copied like any
  other value, and a `-inf` at a position that is not selected must not affect the
  result.
- `lengths`: an integer tensor of shape `(B,)`.
- `padding_side`: the string `"left"` or `"right"`. No other value is passed.
- Returns a float32 tensor of shape `(B, V)`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, T, V)` | float32 |
| `lengths` | `(B,)` | integer |
| result | `(B, V)` | float32 |

## Example

Two rows, `T = 3`, `V = 2`. Each entry is written as `100 b + 10 t + v`, so that it
names its own row, position, and vocabulary index:

```python
logits = [[[  0.0,   1.0], [ 10.0,  11.0], [ 20.0,  21.0]],    # row 0, positions 0, 1, 2
          [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]]]    # row 1, positions 0, 1, 2
lengths = [3, 1]

last_token_logits(logits, lengths, "right")  # [[20.0, 21.0], [100.0, 101.0]]
last_token_logits(logits, lengths, "left")   # [[20.0, 21.0], [120.0, 121.0]]
```

Row 0 is full, so its last real token is at position 2 either way. Row 1 has one real
token. With right padding it sits at position 0, and positions 1 and 2 are padding:
the answer is `[100, 101]`, while `logits[:, -1]` would return `[120, 121]`, the
model's output at a padding position. With left padding the single real token sits at
position 2, the last column.

## Constraints

- The result is compared with the reference exactly: values are copied, not computed.
- $1 \le B, T, V \le 64$, and $1 \le \text{lengths}[b] \le T$.
- No Python loops over batch items are needed.

## Why this matters

Every decoding loop starts with this line: run the prompt, take the logits at the end
of each row, sample. With a single prompt, `logits[:, -1]` is right, and it is what
most tutorials show. With a right-padded batch it reads the model's output at a padding
position for every row shorter than the longest. The longest prompt in the batch
generates correctly and the others produce plausible-looking nonsense, which makes the
bug depend on what a request happened to be batched with. That is the reason
generation code pads on the left: the last real token of every row is then in the last
column, and it stays there as tokens are appended. Scoring and embedding pipelines
usually pad on the right, and they need the gather.
