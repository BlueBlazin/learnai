Generate greedily for a whole batch of prompts at once with a KV cache: one batched
model call per step, however many rows there are. The rows do not finish together.
Each row stops at its own EOS, a finished row must stop collecting tokens while the
batch keeps running for the others, and the loop must end the moment the last row is
done.

## Specification

```python
def batched_greedy(lm, prompts, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), the kit's small
seeded transformer. With batched tokens it works on all rows in one call:

```python
logits, cache = lm(tokens)           # tokens: (B, T), the whole sequences so far
logits, cache = lm(tokens, cache)    # tokens: (B, T), continuing the cached rows
```

- `tokens` is a list of `B` lists of `T` ints (or an integer tensor of shape
  `(B, T)`), `T >= 1`. Every row has the same length.
- `logits` is a float32 tensor of shape `(B, T, vocab_size)`. `logits[b, -1]` scores
  the next token of row `b`.
- `cache` holds the keys and values of all `B` rows. A call with a cache must pass
  the same number of rows. The model never modifies a cache; it returns a new one.
- Rows are independent: what one row is fed never changes another row's logits.
- `lm.counters.forward_calls` counts calls; `lm.counters.tokens_processed` adds
  `B * T` per call.

**What to compute.** Row `b` of the result is exactly what greedy decoding of
`prompts[b]` alone produces: repeatedly take the argmax of the next-token logits,
until the row is *finished*. A row is finished as soon as

- it has generated `eos_id` (**the EOS token is the last element of that row**), or
- it has generated `max_new_tokens` tokens.

Once a row is finished nothing is ever appended to it again, whatever the model
outputs for that row afterwards.

- `prompts`: a list of $B \ge 1$ lists, each with the same number $P \ge 1$ of token
  ids.
- `max_new_tokens`: an int $\ge 1$, the limit for every row.
- `eos_id`: an int or `None`. Only generated tokens count: an `eos_id` inside a prompt
  stops nothing. With `None` every row gets exactly `max_new_tokens` tokens.
- Returns a list of $B$ lists of Python ints: the generated tokens of each row,
  without the prompt. Row $b$ has length $g_b$ with $1 \le g_b \le$ `max_new_tokens`;
  the rows generally have different lengths.

**How to call the model.** Let $g_{\max} = \max_b g_b$ be the length of the longest
row.

1. The first call passes all prompts at once: tokens of shape $(B, P)$.
2. Every later call passes one token per row with the cache: shape $(B, 1)$. A row
   that is still running gets the token it has just generated. **A finished row gets
   `eos_id`** as filler: the batch stays rectangular, and that row's logits are
   ignored from then on.
3. The loop ends as soon as every row is finished. No call is made after that, so the
   last token of the longest row is never fed.

| counter | limit | why |
|---|---|---|
| `forward_calls` | $g_{\max}$ | one call for the prompts, which yields every row's first token, then one per further step of the longest row |
| `tokens_processed` | $B \cdot (P + g_{\max} - 1)$ | every call carries all $B$ rows |

Both limits are what the three rules above produce, so they are exact. Do not shrink
the batch when rows finish, and do not decode the rows one at a time.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompts` | list of $B$ lists of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(B, T, vocab_size)` for a call with tokens `(B, T)` | float32 |
| result | list of $B$ lists; row $b$ has $g_b$ ints | Python `int` |

## Example

```python
lm = TinyLM(fw, seed=3)          # vocab_size 64
prompts = [[0, 52, 7, 13], [0, 16, 13, 51], [0, 55, 38, 4]]

batched_greedy(lm, prompts, 12, eos_id=18)
# [[24, 7, 51, 16, 16, 18],
#  [16, 18],
#  [5, 60, 26, 41, 37, 62, 30, 7, 18]]         9 calls, 36 tokens processed
```

| call | tokens passed | argmax of each row's last logits | finished afterwards |
|---|---|---|---|
| 1 | the prompts, `(3, 4)` | 24, 16, 5 | none |
| 2 | `[[24], [16], [5]]` | 7, **18**, 60 | row 1 |
| 3 | `[[7], [18], [60]]` | 51, 33, 26 | row 1 |
| 4 | `[[51], [18], [26]]` | 16, 56, 41 | row 1 |
| 5 | `[[16], [18], [41]]` | 16, 31, 37 | row 1 |
| 6 | `[[16], [18], [37]]` | **18**, 31, 62 | rows 0, 1 |
| 7 | `[[18], [18], [62]]` | 56, 56, 30 | rows 0, 1 |
| 8 | `[[18], [18], [30]]` | 60, 31, 7 | rows 0, 1 |
| 9 | `[[18], [18], [7]]` | 60, 62, **18** | all |

Row 1 generates EOS in call 2. From call 3 on it is fed 18 and the model keeps
producing output for it (33, 56, 31, ...), none of which belongs to the result. Row 0
finishes in call 6. Row 2 generates EOS in call 9; now every row is finished and the
function returns without feeding anything else. The longest row has 9 tokens, hence
9 calls and $3 \cdot (4 + 8) = 36$ tokens, although the limit was 12 new tokens.

With `max_new_tokens = 7` instead, rows 0 and 1 are unchanged and row 2 is cut to
`[5, 60, 26, 41, 37, 62, 30]`, without an EOS: 7 calls.

## Constraints

- $1 \le B \le 8$, $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens`
  $\le$ `lm.max_context` (128).
- The tokens are compared exactly with greedy decoding of each row on its own, without
  a cache. The test models keep the two largest logits more than `1e-3` apart, so the
  argmax does not depend on how the calls are batched or split. If two logits were
  equal, the argmax is the lower token id (this does not occur in the tests).
- The judge records every call to the model: the first must have shape $(B, P)$, the
  others $(B, 1)$ with the tokens described above, and the two counters must not
  exceed their limits. A solution with the right tokens and the wrong calls fails.

## Why this matters

Batching is where a GPU's throughput comes from: one forward pass over 32 sequences
costs little more than one over a single sequence. But a batch is a rectangle, and
generations are not: some answers take 5 tokens and some 500. The simplest server
loop, static batching, is the one you write here, and it is what Hugging Face
`generate` does: finished rows ride along as padding until the longest row is done.
Its two classic bugs are both about the per-row state. Ending the loop when *any*
row emits EOS truncates everyone else's answer; forgetting that a row has finished
lets the model's continuation past EOS leak into the response. The loop is also
where the waste of static batching becomes visible, in the table above: rows 0 and 1
occupy batch slots for calls that produce nothing. Continuous batching, later in the
curriculum, exists to hand those slots to waiting requests.
