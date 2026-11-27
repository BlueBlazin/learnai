Implement batched greedy decoding with a KV cache using one batched model call per step, regardless of batch size. Rows in a batch do not finish at the same time. Each row must stop collecting tokens once it emits the end-of-sequence (EOS) token or hits the token limit, while the remaining rows keep generating until every row is done.

![This problem is the decode loop, run for a batch of rows at once.](assets/map.svg)

## Specification

```python
def batched_greedy(lm, prompts, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), the kit's small seeded transformer. It processes a batch of `B` rows in a single call:

```python
logits, cache = lm(tokens)           # tokens: (B, T), the whole sequences so far
logits, cache = lm(tokens, cache)    # tokens: (B, T), continuing the cached rows
```

- `tokens` is a list of `B` lists of `T` ints (or an integer tensor of shape `(B, T)`), with `T >= 1`. Every row in a call has the same length `T`.
- `logits` is a float32 tensor of shape `(B, T, vocab_size)`. `logits[b, -1]` scores the next token for row `b`.
- `cache` holds the keys and values for all `B` rows. Any call that passes a cache must pass the same number of rows `B`. The model never mutates a cache in place and always returns a new one.
- Rows are independent. What you feed to one row never affects another row's logits.
- `lm.counters.forward_calls` increments by 1 on each call, and `lm.counters.tokens_processed` increments by `B * T`.

**What to compute.** Row `b` of the output must match greedy decoding of `prompts[b]` on its own. At each step, take the argmax of the next-token logits until the row is *finished*. A row is finished as soon as either of these holds:

- It generates `eos_id` (**the EOS token is included as the last element of that row**).
- It reaches `max_new_tokens` generated tokens.

Once a row is finished, never append to it again, regardless of what the model outputs for that row on later steps.

- `prompts` is a list of $B \ge 1$ lists, each containing the same number $P \ge 1$ of token IDs.
- `max_new_tokens` is an `int` $\ge 1$ giving the generation limit for every row.
- `eos_id` is an `int` or `None`. Only generated tokens trigger stopping, so an `eos_id` inside a prompt has no effect. When `eos_id` is `None`, every row generates exactly `max_new_tokens` tokens.
- Return a list of $B$ lists of Python ints containing only the generated tokens for each row (excluding the prompt). Row $b$ has length $g_b$ with $1 \le g_b \le$ `max_new_tokens`, and different rows may have different lengths.

**How to call the model.** Let $g_{\max} = \max_b g_b$ be the length of the longest generated row.

1. The first call passes all prompts at once as tokens of shape $(B, P)$.
2. Every subsequent call passes one token per row along with the cache, giving shape $(B, 1)$. Each active row receives the token it just generated. **Each finished row receives `eos_id`** as filler so the batch stays rectangular, and its output logits are ignored.
3. The loop stops as soon as every row is finished. Do not make another model call after that, so the final token of the longest row is never fed back into `lm`.

| counter | limit | why |
|---|---|---|
| `forward_calls` | $g_{\max}$ | one call for the prompts, which yields every row's first token, then one per further step of the longest row |
| `tokens_processed` | $B \cdot (P + g_{\max} - 1)$ | every call carries all $B$ rows |

These three rules hit both counter limits exactly. Do not shrink the batch as rows finish, and do not decode rows one at a time.

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

Row 1 generates EOS on call 2. Starting at call 3, it receives `18` as filler and the model keeps producing tokens for it (33, 56, 31, ...), which are discarded. Row 0 finishes on call 6. Row 2 generates EOS on call 9, at which point every row is finished and the function returns without another call. Because the longest row has 9 tokens, the loop makes 9 calls and processes $3 \cdot (4 + 8) = 36$ tokens, even though `max_new_tokens` is 12.

With `max_new_tokens = 7`, rows 0 and 1 are unchanged, and row 2 stops after 7 calls at `[5, 60, 26, 41, 37, 62, 30]` without an EOS.

## Constraints

- $1 \le B \le 8$, $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$ `lm.max_context` (128).
- Generated tokens are compared exactly against uncached single-row greedy decoding for each prompt. The test models keep the top two logits more than `1e-3` apart, so the argmax does not depend on batching or splitting. If two logits were equal, the argmax would pick the lower token ID (this does not occur in the tests).
- The judge records every model call. The first call must have shape $(B, P)$, all later calls must have shape $(B, 1)$ with the tokens specified above, and both counters must stay within their limits. Returning the right tokens with the wrong calls fails.

## Why this matters

GPU throughput depends on batching, because one forward pass over 32 sequences costs little more than one over a single sequence. However, a batch is rectangular while generated answers vary widely in length. Static batching—the loop you implement here and the one used by Hugging Face `generate`—keeps finished rows riding along as padding until the longest row completes.

Its two classic bugs both come from mishandling per-row state. Stopping when *any* row emits EOS truncates every other row's answer. Forgetting that a row has already finished lets the model's continuation past EOS leak into the response. The example table also exposes the main drawback of static batching, since rows 0 and 1 occupy batch slots for steps that produce no useful tokens. Continuous batching, covered later in the curriculum, reclaims those slots for waiting requests.
