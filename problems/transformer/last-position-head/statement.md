Compute the logits for the next token of each sequence in a batch: apply the output
matrix to the hidden state of the last position, and to nothing else. Projecting every
position and discarding all rows but one gives the same numbers for hundreds of times
the work, and the tests time your solution as well as checking it.

## Specification

```python
def next_token_logits(hidden, w_out): ...
```

`hidden` holds the final hidden states of a decoder: `B` sequences of `T` positions,
one vector of width `D` per position. `w_out` is the output (unembedding) matrix that
maps a hidden vector to one logit per vocabulary entry. It is stored in `(out, in)`
layout, shape `(V, D)`: `w_out[i, j]` multiplies component `j` of the hidden vector and
contributes to logit `i`. There is no bias.

The hidden state at position $t$ predicts the token at position $t + 1$, so the token
that follows the sequence is predicted by the **last** position, $T - 1$:

$$
\text{logits}_b = W_{\text{out}}\; h_{b,\,T-1} \in \mathbb{R}^V .
$$

- `hidden`: float32, shape `(B, T, D)`, with $T \ge 1$.
- `w_out`: float32, shape `(V, D)`.
- Returns float32, shape `(B, V)`: row `b` is the logit vector for the token that
  follows sequence `b`. The result has no time axis.

All sequences in the batch have the same length `T`; there is no padding.

## Shapes

| name | shape | dtype |
|---|---|---|
| `hidden` | `(B, T, D)` | float32 |
| `w_out` | `(V, D)` | float32 |
| result | `(B, V)` | float32 |

## Example

Two sequences of two positions, $D = 2$, $V = 4$:

```python
hidden = [[[1.0, 2.0], [3.0, -1.0]], [[0.0, 1.0], [0.5, 0.5]]]  # shape (2, 2, 2)
w_out = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]  # shape (4, 2)

next_token_logits(hidden, w_out)
# [[3.0, -1.0, 2.0, 7.0],
#  [0.5,  0.5, 1.0, 0.5]]
```

The last positions are $[3, -1]$ and $[0.5, 0.5]$. For the first sequence the four
logits are the dot products of $[3, -1]$ with the rows of `w_out`:
$3$, $-1$, $3 - 1 = 2$, and $6 + 1 = 7$. The vectors at position 0 play no part.

Projecting everything would produce a tensor of shape `(2, 2, 4)` whose rows at
position 0, $[1, 2, 3, 0]$ and $[0, 1, 1, -1]$, are computed and then thrown away. Here
that is half of the work. For a prompt of 512 tokens it is 511 parts in 512.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`.
- $1 \le B \le 8$, $1 \le T \le 512$, $1 \le D \le 64$, $1 \le V \le 8192$.
- **Speed.** For `B = 8`, `T = 512`, `D = 64`, `V = 8192`, your function must take at
  most 5 times as long as the reference solution in the same framework. The reference
  performs one matrix product of $B \cdot D \cdot V \approx 4 \times 10^6$
  multiplications. Projecting all positions performs $B \cdot T \cdot D \cdot V
  \approx 2 \times 10^9$ and fills a `(8, 512, 8192)` tensor of 134 MB. The factor of 5
  leaves room for any reasonable way of writing the single product.
- No Python loops are needed.

## Why this matters

When a prompt arrives, the engine runs the whole prompt through the model once
(prefill) and needs exactly one thing from the output layer: the distribution of the
first generated token. The hidden states of the other positions were needed inside the
model, to fill the KV cache, but their logits are predictions of prompt tokens that are
already known. The vocabulary is the widest dimension in the model, tens or hundreds
of thousands of entries, so unembedding a long prompt at every position is a large
matrix product and a logits tensor that can reach gigabytes, all of it discarded. It
is also the default behavior of most model code, which returns logits for every
position because training needs them. Serving code slices the hidden states first.
