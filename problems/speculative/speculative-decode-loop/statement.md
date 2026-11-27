Implement a complete greedy speculative decoding loop. In each round, a small draft model proposes up to `k` tokens using its KV cache, and a target model verifies all of them in **one** forward call. Your loop keeps the agreeing prefix, appends one token from the target model, and rolls both caches back to the verified sequence before starting the next round.

The output must match the target model's own greedy decode token for token. The judge also checks how many forward calls and tokens each model processes.

![This problem is the whole loop, with greedy verification and a cache for each model.](assets/map.svg)

## Specification

```python
def speculative_greedy(target, draft, prompt, k, max_new_tokens, eos_id): ...
```

`target` and `draft` are two [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) models sharing the same vocabulary, created by [`draft_target_pair`](../../../python/src/learnai/kit/draft.py). Both use the same KV-cache interface as in `cached-greedy-decode`:

```python
logits, cache = lm(tokens)           # no cache: `tokens` is the whole sequence so far
logits, cache = lm(tokens, cache)    # with a cache: `tokens` continue the cached prefix
short = cache.truncate(n)            # a new cache covering only the first n tokens
```

When called with `T` tokens, a model returns `logits` of shape `(T, vocab_size)`, where row `i` scores the token after `tokens[i]`. Caches are immutable. Calling a model or calling `truncate(n)` returns a new cache (with its length exposed as `cache.length`) and leaves the old cache unchanged. Each model tracks `forward_calls` and `tokens_processed` in its `counters` attribute.

**Inputs and output.**
- `prompt` is a non-empty list of $P \ge 1$ token IDs.
- `k` is the draft block length ($\ge 1$).
- `max_new_tokens` is the maximum number of tokens to generate ($\ge 1$).
- `eos_id` is an integer stop token ID, or `None` to disable early stopping.
- Return the generated continuation (excluding `prompt`) as a list of $g$ Python `int`s, where $1 \le g \le \texttt{max\_new\_tokens}$.

Each generated token must equal the argmax of the target model's next-token logits given `prompt` and all previously generated tokens. Generation stops once `max_new_tokens` tokens have been produced, or immediately after generating `eos_id` (including `eos_id` as the last returned token). An `eos_id` inside `prompt` does not trigger stopping. The draft model never changes the output tokens, only the number of target calls needed.

**Round procedure.** Let $S$ be the verified sequence (`prompt` plus all tokens generated so far) and let $r$ be the number of tokens still remaining in the `max_new_tokens` budget. Each *round* runs four steps:

1. **Draft.** Choose $k' = \min(k,\, r - 1)$. The draft model proposes $k'$ tokens $d_1, \dots, d_{k'}$ greedily and autoregressively, where $d_j$ is the argmax of the draft's next-token logits given $S, d_1, \dots, d_{j-1}$. When $r = 1$, no tokens are drafted ($k' = 0$).
2. **Verify.** Run one target forward call to obtain the target's greedy tokens $g_0, \dots, g_{k'}$ at all $k' + 1$ positions, where $g_0$ follows $S$ and $g_j$ follows $S, d_1, \dots, d_j$.
3. **Accept.** Let $n$ be the length of the longest matching prefix where $d_j = g_{j-1}$ for all $1 \le j \le n$. Emit $d_1, \dots, d_n$ followed by $g_n$. When $n < k'$, $g_n$ is the target's correction at the first mismatch. When $n = k'$, $g_n$ is the bonus token. A round always emits $n + 1$ tokens, and because $k' \le r - 1$, it never exceeds $r$.
4. **Stop or roll back.** If `eos_id` appears in the round's emitted tokens, truncate the emitted tokens right after the first `eos_id` and return. If `max_new_tokens` tokens have been generated, return. Otherwise, truncate both caches so they contain only verified tokens (no rejected drafts) and begin the next round.

Because every round emits $g_n$ directly from the target, the loop always advances by at least one token even if every draft is rejected.

**Cost limits.** Let $R$ be the number of rounds the procedure above takes for the given models and arguments.

| counter | limit | why |
|---|---|---|
| target `forward_calls` | $R$ | one verification call per round |
| target `tokens_processed` | $P + R\,(k + 1)$ | the prompt once, plus at most $k$ drafts and one unseen verified token per round |
| draft `tokens_processed` | $P + R\,(k + 1)$ | the prompt once, plus at most $k + 1$ tokens per round |

Any implementation that returns the exact tokens within these limits passes. Re-feeding the full sequence every round exceeds the token limits. Verifying fewer drafts than allowed, or drafting from a cache that still holds rejected tokens, increases the number of rounds above $R$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from either model) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| result | list of $g$ ints | Python `int` |

## Example

```python
draft, target = draft_target_pair(fw, seed=3, agreement=0.9)

speculative_greedy(target, draft, [0, 17, 42, 5], 4, 12, None)
# [42, 62, 33, 33, 40, 32, 45, 37, 40, 33, 40, 55]      4 target calls

speculative_greedy(target, draft, [0, 17, 42, 5], 4, 12, 40)
# [42, 62, 33, 33, 40]                                   2 target calls
```

Here is the first call round by round ($P = 4$, $k = 4$):

| round | $r$ | drafts | target's tokens $g_0 \dots g_{k'}$ | $n$ | generated |
|---|---|---|---|---|---|
| 1 | 12 | `33 40 33 33` | `42` ... | 0 | `42` |
| 2 | 11 | `62 33 33 40` | `62 33 33 40 32` | 4 | `62 33 33 40 32` |
| 3 | 6 | `45 37 40 33` | `45 37 40 33 40` | 4 | `45 37 40 33 40` |
| 4 | 1 | (none) | `55` | 0 | `55` |

In round 1, the first draft token is wrong because the target predicts 42. One token is emitted, and the three later drafts are discarded regardless of what the target predicted for them. Both caches now include rejected tokens and must be rolled back. Rounds 2 and 3 accept all four drafts and earn a bonus token each. In round 4, only one token remains in the budget, so nothing is drafted and the target generates token 55 directly.

![Round 1 of the first call, step by step. The target's first token sits under the draft it is compared with.](assets/round-steps.svg)

Here is what the reference loop feeds to each model, along with the cache length before each call:

| round | target call (cache) | draft calls (cache before the first) | target cache kept |
|---|---|---|---|
| 1 | `0 17 42 5 33 40 33 33` (none) | `0 17 42 5`, `33`, `40`, `33` (none) | 4 of 8 |
| 2 | `42 62 33 33 40` (4) | `42`, `62`, `33`, `33` (4) | 9 of 9 |
| 3 | `32 45 37 40 33` (9) | `40 32`, `45`, `37`, `40` (8) | 14 of 14 |
| 4 | `40` (14) | | |

The target processes $8 + 5 + 5 + 1 = 19$ tokens across 4 calls (limit $4 + 4 \times 5 = 24$), and the draft processes $7 + 4 + 5 = 16$ tokens (limit 24). Plain cached greedy decoding would take 12 target calls for the same 12 tokens. In round 3, the draft cache is one token shorter than the target cache because the draft model never fed its own last proposal `40` during round 2.

In the second call, token 40 is the fourth draft of round 2 and is accepted. Because it equals `eos_id`, generation stops there and the bonus token 32 is not emitted.

## Constraints

- $1 \le P$, $1 \le k \le 8$, $1 \le \texttt{max\_new\_tokens}$, and $P + \texttt{max\_new\_tokens} + k \le \texttt{target.max\_context}$ (128 for both models).
- Output tokens are compared exactly against the target model's greedy decode. Wherever either model takes an argmax, its two largest logits differ by at least `1e-3`. Splitting calls differently changes logits by only about `1e-5`, so it cannot alter the tokens or the round count.
- All three counters are checked against the cost limits above, where $R$ is computed by running the round procedure. Exceeding any limit fails even if the output tokens are exact.
- Argmax ties break toward the lower token ID (ties do not occur in the tests).

## Why this matters

This loop is the core of speculative decoding in inference servers such as llama.cpp, vLLM, and Hugging Face `transformers` assisted generation. The single-step verification rule is straightforward, but the loop is where most engineering bugs happen because two KV caches advance at different rates and both must rewind after every rejection.

A cache that retains a rejected token does not raise an error. On the target side, it silently produces fluent text that diverges from the target's greedy decode. On the draft side, the output stays exact while the acceptance rate collapses, turning a speedup into a slowdown that only benchmarks expose. Recomputing either cache from scratch instead of truncating it wastes prefill work and erases the gains of speculation.
