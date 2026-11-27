Apply the CTRL-style repetition penalty to a batch of next-token logits using vectorized
tensor operations. For each row, push the logit of every distinct non-padding token in
that row's history toward a lower probability, penalizing each seen token once regardless
of how many times it occurred.

![This problem is Penalties, here the repetition penalty on tokens already in the history.](assets/map.svg)

## Specification

```python
def apply_repetition_penalty(logits, prev_tokens, penalty, pad_id): ...
```

`logits` holds unnormalized next-token scores for a batch of `B` sequences over a
vocabulary of `V` tokens. `prev_tokens` holds each sequence's previously generated token
IDs, padded to a common length `T` with `pad_id`. Both `penalty` (a Python float
$\theta \ge 1$) and `pad_id` (a Python int in $[0, V)$) are shared across the batch.

For each row $b$ ($0 \le b < B$), let

$$
S_b = \{\, \text{prev\_tokens}[b, t] : 0 \le t < T \,\} \setminus \{\text{pad\_id}\}
$$

be the set of distinct non-padding tokens in that row's history. Write $x = \text{logits}$.
To make seen tokens less likely, positive logits are divided by $\theta$ and non-positive
logits are multiplied by $\theta$:

$$
\text{out}[b, v] =
\begin{cases}
x[b, v] \,/\, \theta & \text{if } v \in S_b \text{ and } x[b, v] > 0, \\
x[b, v] \cdot \theta & \text{if } v \in S_b \text{ and } x[b, v] \le 0, \\
x[b, v] & \text{if } v \notin S_b .
\end{cases}
$$

Follow these rules:

- Entries of `logits` may be `-inf` (masked tokens), and any `-inf` entry must stay
  `-inf`.
- When $\theta = 1$, all logits remain unchanged.
- **Every entry of `prev_tokens` equal to `pad_id` is padding**, whether it appears at
  the start, middle, or end of a row. Padding entries are ignored, so the logit of
  `pad_id` is never penalized. A row may consist of padding only.
- Only set membership in $S_b$ matters. A token that appears five times in the history
  is penalized the same amount as one that appears once.
- Rows are independent. Row $b$ of the result depends only on row $b$ of `logits` and
  `prev_tokens`.
- Return a new tensor without modifying the inputs.

Do not loop in Python over batch rows, history positions, or vocabulary entries. The
speed constraint below enforces this.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `prev_tokens` | `(B, T)` | integer |
| result | `(B, V)` | float32 |

## Example

```python
logits = [[2.0, -1.0, 0.5, -3.0, 0.0,  4.0],
          [1.0,  1.0, -2.0, -2.0, 3.0, -0.5]]
prev_tokens = [[5, 1, 5, 0],
               [2, 4, 0, 0]]

apply_repetition_penalty(logits, prev_tokens, penalty=1.25, pad_id=0)
# [[2.0, -1.25, 0.5, -3.0, 0.0,  3.2],
#  [1.0,  1.0, -2.5, -2.0, 2.4, -0.5]]
```

Row 0 has seen tokens 5 and 1, with token 5 appearing twice alongside one padding entry
(`0`). Token 5 has a positive logit, so it is divided once to give $4.0 / 1.25 = 3.2$.
Token 1 has a negative logit, so it is multiplied to give $-1.0 \times 1.25 = -1.25$.
Dividing it instead would give $-0.8$, making the token *more* likely. Token 0 stays at
`2.0` because `0` is `pad_id`, not a generated token.

Row 1 has seen tokens 2 and 4, giving $-2.0 \times 1.25 = -2.5$ and $3.0 / 1.25 = 2.4$.
Token 3 also has logit $-2.0$ but is not in row 1's history, so it is left unchanged.
Tokens 5 and 1 are penalized in row 0 only.

## Constraints

- $1 \le B \le 4096$, $1 \le T \le 64$, $2 \le V \le 1000$, and $1 \le \theta \le 2$. All
  entries of `prev_tokens` lie in $[0, V)$.
- Results are compared against the reference with a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. Any `-inf` entry must stay `-inf`.
- **Speed.** On a batch with $B = 4096$, $T = 16$, and $V = 256$, your solution must
  take at most **8 times** as long as the reference in the same framework, measured in
  the same process (interleaved runs, median of 7). The reference performs one scatter
  and two selects, doing work proportional to $B \cdot (T + V)$. Any vectorized
  formulation with similar cost passes with a wide margin, while a Python loop over the
  4096 rows is roughly 100 times slower and fails.

## Why this matters

Logit processors run for every sequence in a batch at every decoding step, between the
forward pass and the sampler. A server decoding hundreds of sequences per step cannot
afford a Python loop there. At tens of microseconds per row, loop overhead alone can
cost more than the sampler itself.

The edge cases matter as much as the speed. Dividing every seen logit regardless of sign
rewards repeated tokens whose logits are negative. Counting padding as history
suppresses whichever real token shares `pad_id` (often the end-of-sequence token, which
causes generations to run on). Sharing one seen set across the batch leaks one request's
history into every other request's output.
