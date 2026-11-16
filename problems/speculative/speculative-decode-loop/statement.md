Implement a complete greedy speculative decoding loop: a small draft model proposes
`k` tokens through its KV cache, the target model checks all of them in **one**
call, the agreeing prefix is kept, and both caches are rolled back to the verified
sequence before the next round. The result must be token for token what the target
model's own greedy decode produces, and the judge counts the work both models were
asked to do.

## Specification

```python
def speculative_greedy(target, draft, prompt, k, max_new_tokens, eos_id): ...
```

`target` and `draft` are two [`TinyLM`](../../../python/src/learnai/kit/tinylm.py)
models over the same vocabulary, built by
[`draft_target_pair`](../../../python/src/learnai/kit/draft.py). Each is called as
in `cached-greedy-decode`:

```python
logits, cache = lm(tokens)           # no cache: `tokens` is the whole sequence so far
logits, cache = lm(tokens, cache)    # with a cache: `tokens` continue the cached prefix
short = cache.truncate(n)            # a new cache covering only the first n tokens
```

`logits` has one row per token passed in that call, shape `(T, vocab_size)`, and row
`i` scores the token after `tokens[i]`. Caches are immutable: a call or a `truncate`
returns a new cache (`cache.length` tokens long) and leaves the old one valid. Each
model counts `forward_calls` and `tokens_processed` in its own `counters`.

**What to return.** The target's greedy continuation of `prompt`: each token is the
argmax of the target's next-token logits given the prompt and the tokens before it.
Generation stops after `max_new_tokens` tokens, or right after a generated token
equal to `eos_id` (that token is the last element of the result; `None` disables
this; an `eos_id` inside the prompt stops nothing). The draft model never changes
the result, only how many target calls it takes.

**How to get there.** Write $S$ for the verified sequence (the prompt followed by
the tokens generated so far) and $r$ for the number of tokens still allowed. One
*round*:

1. **Draft.** The draft model proposes $k' = \min(k,\, r - 1)$ tokens
   $d_1, \dots, d_{k'}$ greedily and autoregressively: $d_j$ is the argmax of the
   draft's next-token logits given $S, d_1, \dots, d_{j-1}$.
2. **Verify.** One target call yields the target's greedy token $g_j$ at each of
   the $k' + 1$ positions: $g_0$ given $S$, and $g_j$ given $S, d_1, \dots, d_j$.
3. **Accept.** Let $n$ be the length of the longest prefix with $d_j = g_{j-1}$ for
   all $j \le n$. The round generates $d_1, \dots, d_n$ followed by $g_n$: the
   target's correction at the first mismatch, or its bonus token when all $k'$
   drafts were accepted. That is $n + 1$ tokens, never more than $r$.
4. **Stop or roll back.** If one of the round's tokens equals `eos_id`, the result
   ends with the first such token; the round's later tokens are dropped. If
   `max_new_tokens` is reached, stop. Otherwise bring both caches back to verified
   tokens only (rejected drafts must not remain in either) and start the next round.

Because a round always generates $g_n$, a token the target chose itself, the loop
makes progress even when every draft is wrong, and with $r = 1$ nothing is drafted.

- `prompt`: a list of $P \ge 1$ ints.
- `k`: the draft length, an int $\ge 1$.
- `max_new_tokens`: an int $\ge 1$.
- `eos_id`: an int or `None`.
- Returns the generated tokens (without the prompt) as a list of $g$ Python ints,
  $1 \le g \le$ `max_new_tokens`.

**Cost limits.** Let $R$ be the number of rounds the procedure above takes for the
given models and arguments.

| counter | limit | why |
|---|---|---|
| target `forward_calls` | $R$ | one verification per round |
| target `tokens_processed` | $P + R\,(k + 1)$ | the prompt once; then, per round, at most $k$ drafts and the one verified token the target has not seen |
| draft `tokens_processed` | $P + R\,(k + 1)$ | the prompt once; then, per round, at most $k + 1$ tokens |

Any loop that returns the right tokens within these limits is accepted. Feeding a
model its whole sequence again in every round exceeds the token limits; verifying
fewer drafts per round than possible, or drafting from a cache that still contains
rejected tokens, needs more rounds than $R$.

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

The first call, round by round ($P = 4$, $k = 4$):

| round | $r$ | drafts | target's tokens $g_0 \dots g_{k'}$ | $n$ | generated |
|---|---|---|---|---|---|
| 1 | 12 | `33 40 33 33` | `42` ... | 0 | `42` |
| 2 | 11 | `62 33 33 40` | `62 33 33 40 32` | 4 | `62 33 33 40 32` |
| 3 | 6 | `45 37 40 33` | `45 37 40 33 40` | 4 | `45 37 40 33 40` |
| 4 | 1 | (none) | `55` | 0 | `55` |

In round 1 the draft's first guess is wrong: the target says 42. One token is
generated, and the three later drafts are void, whatever the target's rows for them
say. Both caches now cover rejected tokens and are rolled back. Rounds 2 and 3 are
accepted in full and earn a bonus token each. In round 4 a single token is left, so
nothing is drafted and the target produces it alone.

What the reference loop feeds each model, with the cache length before the call:

| round | target call (cache) | draft calls (cache before the first) | target cache kept |
|---|---|---|---|
| 1 | `0 17 42 5 33 40 33 33` (none) | `0 17 42 5`, `33`, `40`, `33` (none) | 4 of 8 |
| 2 | `42 62 33 33 40` (4) | `42`, `62`, `33`, `33` (4) | 9 of 9 |
| 3 | `32 45 37 40 33` (9) | `40 32`, `45`, `37`, `40` (8) | 14 of 14 |
| 4 | `40` (14) | | |

The target processes $8 + 5 + 5 + 1 = 19$ tokens in 4 calls (limit
$4 + 4 \times 5 = 24$) and the draft $7 + 4 + 5 = 16$ (limit 24). Plain cached
greedy decoding needs 12 target calls for the same tokens. Note round 3: the draft
never fed its own last proposal `40` in round 2, so after a fully accepted round its
cache is one token further behind than the target's.

In the second call, token 40 is the fourth draft of round 2 and is accepted. It
equals `eos_id`, so the result ends there and the bonus token 32 is not emitted.

## Constraints

- $1 \le P$, $1 \le k \le 8$, $1 \le$ `max_new_tokens`, and
  $P +$ `max_new_tokens` $+\, k \le$ `target.max_context` (128, the same for both
  models).
- Tokens are compared exactly with the target's greedy decode. Wherever a model's
  argmax is used, in the target or in the draft, its two largest logits differ by at
  least `1e-3`, so the way calls are split (which changes logits by about `1e-5`)
  affects neither the tokens nor the number of rounds.
- The three counters are compared with the limits above; $R$ is computed by running
  the round procedure. A solution over a limit fails even when its tokens are right.
- Ties in an argmax go to the lower token id (this does not occur in the tests).

## Why this matters

This loop is what "speculative decoding" means in an inference server: a draft
model next to the main model in llama.cpp, the draft-model and n-gram speculators
in vLLM, assisted generation in `transformers`. The single-step rule is easy; the loop is
where the engineering is, because two KV caches advance at different speeds and
both must be rewound after every rejection. A cache that keeps a rejected token does
not crash. On the target side it yields fluent output that is no longer the target's
greedy decode; on the draft side the output stays correct while the acceptance rate
collapses and the "speedup" turns into a slowdown that only a benchmark reveals.
And recomputing either cache instead of truncating it gives away, in prefill work,
everything the speculation gained.
