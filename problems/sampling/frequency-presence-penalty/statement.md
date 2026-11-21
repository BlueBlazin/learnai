Implement the frequency and presence penalties of the OpenAI API for a batch of
sequences. They are additive penalties on the logits of tokens the model has already
produced, and they come in two parts: the count of what was generated, and the
formula that uses it. Both parts have a detail that is easy to get wrong: the count
covers generated tokens only, not the prompt, and the two penalties use the count
differently.

![This problem is Penalties, here the two additive penalties on tokens already generated.](assets/map.svg)

## Specification

```python
def count_generated(tokens, prompt_lengths, vocab_size): ...
def apply_penalties(logits, token_counts, frequency_penalty, presence_penalty): ...
```

**`count_generated`.** Row $b$ of `tokens` is one sequence: a prompt of
`prompt_lengths[b]` tokens followed by the tokens the model has generated so far. The
result counts, per row, how often each vocabulary entry occurs among the generated
positions:

$$
\text{counts}[b, v] = \#\{\, t : t \ge \text{prompt\_lengths}[b]
\ \text{ and } \ \text{tokens}[b, t] = v \,\}.
$$

- `tokens`: an integer tensor of shape `(B, T)` with values in $[0, \text{vocab\_size})$.
  All rows have the same total length `T`; there is no padding.
- `prompt_lengths`: an integer tensor of shape `(B,)` with
  $0 \le \text{prompt\_lengths}[b] \le T$. Rows have different prompt lengths. With
  `0` every position was generated; with `T` none was, and the row's counts are all 0.
- `vocab_size`: a Python int, `V`.
- Returns an integer tensor of shape `(B, V)`.

Prompt tokens are never counted, even when the same token is generated later: a token
that occurs twice in the prompt and three times after it has count 3.

**`apply_penalties`.** With $c$ = `token_counts`, $\alpha_f$ = `frequency_penalty`, and
$\alpha_p$ = `presence_penalty`:

$$
\text{out}[b, v] = \text{logits}[b, v] \;-\; c[b, v] \cdot \alpha_f \;-\;
\mathbb{1}[c[b, v] > 0] \cdot \alpha_p .
$$

The frequency penalty is charged once per occurrence. The presence penalty is charged
once per token, however often it occurred. Both are subtracted, whatever the sign of
the logit.

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (masked); a
  masked logit stays `-inf`.
- `token_counts`: an integer tensor of shape `(B, V)` with non-negative entries, as
  returned by `count_generated`.
- `frequency_penalty`, `presence_penalty`: Python floats, the same for every row.
  Either may be zero or negative; a negative penalty raises the logits of repeated
  tokens.
- Returns a new float32 tensor of shape `(B, V)`.

The two functions are tested separately and together.

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

Row 0 has a prompt of two tokens, `[3, 2]`, and has generated `[5, 5, 2, 5]`. Token 5
was generated three times and token 2 once. Token 3 occurs only in the prompt, so its
count is 0; token 2 occurs in the prompt as well, and only its generated occurrence
counts. Token 5 loses $3 \times 0.5 + 1.0 = 2.5$ (from 3.0 to 0.5) and token 2 loses
$1 \times 0.5 + 1.0 = 1.5$ (from 2.0 to 0.5).

Row 1 has a prompt of three tokens, `[1, 1, 4]`, and has generated `[0, 4, 4]`. Token
1 fills two thirds of the prompt and is not penalized: its logit stays 4.0. Token 4
loses $2 \times 0.5 + 1.0 = 2.0$ and token 0 loses 1.5.

## Constraints

- $1 \le B \le 64$, $1 \le T \le 128$, $1 \le V \le 1000$. Both penalties lie in
  $[-2, 2]$, the range the OpenAI API accepts. Finite logits lie in $[-100, 100]$.
- Counts are compared with the reference exactly.
- Penalized logits are compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`; `-inf` must stay `-inf`, and a NaN is always wrong.
- No Python loop over rows, positions, or vocabulary entries is needed.

## Why this matters

`frequency_penalty` and `presence_penalty` are fields of the OpenAI chat completions
API, so every OpenAI-compatible server has to implement them, and clients expect the
OpenAI semantics. The two knobs do different jobs: the frequency penalty keeps
growing and stops a loop on one token; the presence penalty is a flat, one-time cost
that nudges the model toward tokens it has not used yet. Collapse them into one and
neither setting means what the documentation says. Counting the prompt is the subtler
failure: a long document in the prompt then suppresses its own vocabulary, and the
model cannot quote or summarize the text it was given, with a strength that depends
on the prompt and not on anything the model did.
