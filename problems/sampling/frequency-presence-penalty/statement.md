Implement the frequency and presence penalties of the OpenAI API for a batch of
sequences. Write `count_generated` to count how often each token appears in the
generated portion of each sequence (excluding the prompt), and `apply_penalties` to
subtract both additive penalties from the logits.

![This problem is Penalties, here the two additive penalties on tokens already generated.](assets/map.svg)

## Specification

```python
def count_generated(tokens, prompt_lengths, vocab_size): ...
def apply_penalties(logits, token_counts, frequency_penalty, presence_penalty): ...
```

Consider a batch of `B` sequences, each of total length `T`, over a vocabulary of size
`V` (`vocab_size`, a Python `int`).

**`count_generated`.** Row $b$ of `tokens` holds a prompt of `prompt_lengths[b]` tokens
followed by the tokens generated so far. Token IDs lie in $[0, V)$, and every row has
the same total length `T` with no padding. Prompt lengths vary by row and satisfy
$0 \le \text{prompt\_lengths}[b] \le T$. A prompt length of `0` means every position was
generated. A prompt length of `T` means no tokens were generated, so that row's counts
are all `0`.

For each row $b$ and vocabulary token $v$, count how often $v$ appears at generated
positions:

$$
\text{counts}[b, v] = \#\{\, t : t \ge \text{prompt\_lengths}[b]
\ \text{ and } \ \text{tokens}[b, t] = v \,\}.
$$

Prompt tokens are never counted, even when the same token is generated later. A token
that appears twice in the prompt and three times in the generated suffix has a count
of `3`.

**`apply_penalties`.** Let $c = \texttt{token\_counts}$ be a non-negative integer count
tensor as returned by `count_generated`, $\alpha_f = \texttt{frequency\_penalty}$, and
$\alpha_p = \texttt{presence\_penalty}$. Both penalties are Python floats shared across
all rows. The penalized logits are:

$$
\text{out}[b, v] = \text{logits}[b, v] - c[b, v] \cdot \alpha_f -
\mathbb{1}[c[b, v] > 0] \cdot \alpha_p.
$$

The frequency penalty is charged once per generated occurrence. The presence penalty is
charged once per token that appeared at least once, regardless of how many times it
occurred. Both penalties are subtracted regardless of the sign of the logit.

Either penalty may be zero or negative. A negative penalty raises the logits of repeated
tokens. Entries in `logits` may be `-inf` (masked), and masked logits must remain
`-inf`. Return a new float32 tensor.

The two functions are tested both separately and together.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(B, T)` | integer |
| `prompt_lengths` | `(B,)` | integer |
| result of `count_generated` | `(B, V)` | integer |
| `logits` | `(B, V)` | float32 |
| `token_counts` | `(B, V)` | integer |
| result of `apply_penalties` | `(B, V)` | float32 |

## Example

```python
tokens = [[3, 2, 5, 5, 2, 5], [1, 1, 4, 0, 4, 4]]
prompt_lengths = [2, 3]

counts = count_generated(tokens, prompt_lengths, 6)
# [[0, 0, 1, 0, 0, 3],
#  [1, 0, 0, 0, 2, 0]]

logits = [[1.0, 0.0, 2.0, -1.0, 0.5, 3.0], [2.0, 4.0, -inf, 0.0, 1.0, -2.0]]

apply_penalties(logits, counts, frequency_penalty=0.5, presence_penalty=1.0)
# [[1.0, 0.0,  0.5, -1.0,  0.5,  0.5],
#  [0.5, 4.0, -inf,  0.0, -1.0, -2.0]]
```

Row 0 has a prompt of two tokens (`[3, 2]`) and generated `[5, 5, 2, 5]`. Token 5 was
generated three times and token 2 once. Token 3 appears only in the prompt, so its count
is `0`. Token 2 also appears in the prompt, but only its generated occurrence counts.
Token 5 loses $3 \times 0.5 + 1.0 = 2.5$ (from `3.0` to `0.5`), and token 2 loses
$1 \times 0.5 + 1.0 = 1.5$ (from `2.0` to `0.5`).

Row 1 has a prompt of three tokens (`[1, 1, 4]`) and generated `[0, 4, 4]`. Token 1
appears only in the prompt, so it is not penalized and its logit stays `4.0`. Token 4
loses $2 \times 0.5 + 1.0 = 2.0$ (from `1.0` to `-1.0`), and token 0 loses
$1 \times 0.5 + 1.0 = 1.5$ (from `2.0` to `0.5`). Token 2 starts at `-inf` and remains
`-inf`.

## Constraints

- $1 \le B \le 64$, $1 \le T \le 128$, and $1 \le V \le 1000$.
- Both penalties lie in $[-2, 2]$, matching the range accepted by the OpenAI API.
- Finite logits lie in $[-100, 100]$.
- Counts are compared against the reference exactly.
- Penalized logits are compared with a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`. Entries equal to `-inf` must stay `-inf`, and any `NaN` is
  always wrong.
- You do not need Python loops over rows, positions, or vocabulary entries.

## Why this matters

`frequency_penalty` and `presence_penalty` are standard parameters in the OpenAI chat
completions API, so OpenAI-compatible servers must implement their exact semantics. The
two knobs serve different roles. The frequency penalty grows with every repetition and
breaks loops on a single token. The presence penalty applies a flat one-time cost that
nudges the model toward tokens it has not used yet. Collapsing them into one term
breaks both behaviors.

Counting prompt tokens causes a subtler failure. When a long document sits in the
prompt, penalizing prompt tokens suppresses the document's own vocabulary. The model
then struggles to quote or summarize its input, with a penalty strength driven by the
prompt rather than by what the model generated.
