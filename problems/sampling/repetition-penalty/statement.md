Apply the CTRL-style repetition penalty to a batch of next-token logits: every token
that already occurs in a row's history has its logit pushed toward "less likely".
The arithmetic is one line. The exercise is to do it for the whole batch with tensor
operations, without a Python loop over rows or tokens, while ignoring padding and
penalizing each token once however often it occurred.

## Specification

```python
def apply_repetition_penalty(logits, prev_tokens, penalty, pad_id): ...
```

Row $b$ of `prev_tokens` is the history of sequence $b$: the token ids it has
generated so far, padded to a common length $T$ with `pad_id`. Let

$$
S_b = \{\, \text{prev\_tokens}[b, t] : 0 \le t < T \,\} \setminus \{\text{pad\_id}\}
$$

be the set of distinct non-padding tokens in that row. With $\theta$ = `penalty` and
$x$ = `logits`, the result is

$$
\text{out}[b, v] =
\begin{cases}
x[b, v] \,/\, \theta & \text{if } v \in S_b \text{ and } x[b, v] > 0, \\
x[b, v] \cdot \theta & \text{if } v \in S_b \text{ and } x[b, v] \le 0, \\
x[b, v] & \text{if } v \notin S_b .
\end{cases}
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (masked); a
  masked logit stays `-inf`.
- `prev_tokens`: an integer tensor of shape `(B, T)` with values in $[0, V)$.
- `penalty`: a Python float, $\theta \ge 1$. With $\theta = 1$ the logits are unchanged.
- `pad_id`: a Python int in $[0, V)$. **Every entry of `prev_tokens` equal to `pad_id`
  is padding**, wherever it is in the row (start, middle, or end), and is ignored: the
  logit of `pad_id` is never penalized. A row may consist of padding only.
- Membership in $S_b$ is all that matters. A token that occurs five times in the
  history is penalized exactly as much as one that occurs once.
- Each row uses its own history. Row $b$ of the result depends only on row $b$ of the
  inputs.
- Returns a new float32 tensor of shape `(B, V)`. Do not modify the inputs.

Do not loop in Python over batch rows, history positions, or vocabulary entries. The
speed case below enforces it.

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

Row 0 has seen tokens 5 and 1 (token 5 twice, and one padding entry). Token 5 has a
positive logit, so it is divided: $4.0 / 1.25 = 3.2$, once, not twice. Token 1 has a
negative logit, so it is multiplied: $-1.0 \times 1.25 = -1.25$. Dividing it would
give $-0.8$, which makes the token *more* likely. The logit of token 0 stays `2.0`:
`0` is the pad id here, not a generated token.

Row 1 has seen tokens 2 and 4: $-2.0 \times 1.25 = -2.5$ and $3.0 / 1.25 = 2.4$. Token
3 also has logit $-2.0$ but is not in row 1's history, and tokens 5 and 1 are
penalized in row 0 only.

## Constraints

- $1 \le B \le 4096$, $1 \le T \le 64$, $2 \le V \le 1000$, $1 \le \theta \le 2$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`; `-inf` must stay `-inf`.
- **Speed.** On a batch with $B = 4096$, $T = 16$, $V = 256$, the solution must take at
  most **8 times** as long as the reference written in the same framework, measured in
  the same process (interleaved runs, median of 7). The reference does one scatter and
  two selects, work proportional to $B \cdot (T + V)$. A different vectorized
  formulation of that cost passes with a wide margin; a Python loop over the 4096 rows
  is roughly 100 times slower than the reference and fails.

## Why this matters

Logit processors run for every sequence in the batch at every decoding step, on the
serving path between the forward pass and the sampler. A server decoding hundreds of
sequences per step cannot afford a Python-level loop there: at tens of microseconds
per row, the penalty would cost more than the sampler it feeds. The details matter as
much as the speed. The original formula divides every seen logit, which rewards
repeated tokens whose logits are negative; counting padding as history suppresses
whichever real token shares the pad id (often end-of-sequence, which makes generations
run on); and sharing the seen set across the batch leaks one request's history into
another's output.
