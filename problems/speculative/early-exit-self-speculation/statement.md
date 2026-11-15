Implement self-speculative decoding: the model drafts for itself. Its first `E`
layers and the shared output head propose `k` tokens, its remaining layers check
all of them in **one** call, and, because draft and verification are two halves of
the same forward pass, nothing the early layers computed is computed again. The
result must be token for token the full model's greedy decode, and the judge counts
which layers ran on how many tokens.

## Specification

```python
def self_speculative_greedy(lm, prompt, exit_layer, k, max_new_tokens): ...
```

`lm` is a [`LayeredLM`](../../../python/src/whetstone/kit/layeredlm.py): a decoder
with $L$ = `lm.n_layers` layers whose stack can be run one range at a time.

```python
h = lm.embed(tokens, offset=n)  # (T, d): tokens at positions n, n + 1, ...
h, cache = lm.layers(h, a, b, cache)  # layers a .. b - 1 on these positions
logits = lm.head(lm.norm(h))  # (T, vocab_size): the output head, after the final norm
cache = lm.empty_cache()  # no layer has seen a token
cache = cache.truncate(n)  # every layer rolled back to its first n tokens
```

- `lm.layers(h, a, b, cache)` runs layers `a` to `b - 1` on hidden states `h` and
  returns the new states and a new cache (caches are immutable). The rows of `h`
  continue what those layers have cached: they are the states of the **next**
  positions, in order, and the layers `a..b-1` must all have cached the same number
  of tokens. Early layers may be ahead of late layers; `cache.lengths` lists the
  tokens cached per layer and `cache.length` is their maximum.
- The model's logits for a sequence are
  `lm.head(lm.norm(states after layer L - 1))`. An *early exit* after layer
  `E - 1` applies the **same** norm and head to the states after layer `E - 1`.
  The final norm has a learned gain: `lm.head(h)` without `lm.norm` is a different
  function with a different argmax.
- `lm.counters` counts `layer_calls` (one per layer per call of `lm.layers`,
  whatever the number of tokens) and `layer_tokens` (layers times tokens).
  `embed`, `norm`, and `head` are free.

**What to return.** The full model's greedy continuation of `prompt`:
`max_new_tokens` tokens, each the argmax of the full model's next-token logits
given the prompt and the tokens before it.

**How to get there.** Write $S$ for the verified sequence (the prompt followed by
the tokens generated so far), $E$ for `exit_layer`, and $r$ for the number of
tokens still to generate. One *round*:

1. **Draft with layers $0..E-1$.** Propose $k' = \min(k,\, r - 1)$ tokens
   $d_1, \dots, d_{k'}$: $d_j$ is the argmax of the early exit's logits at the last
   position of $S, d_1, \dots, d_{j-1}$. The early layers run on every token they
   have not processed: in the first round the whole prompt, in later rounds the
   newest token of $S$, and then the drafts one by one, including $d_{k'}$ (whose
   exit state the verification needs).
2. **Verify with layers $E..L-1$.** One call of `lm.layers(..., E, L, ...)` on the
   exit states of exactly those positions gives the full model's greedy token at
   each of the last $k' + 1$ of them: $g_0$ given $S$, and $g_j$ given
   $S, d_1, \dots, d_j$.
3. **Accept.** Let $n$ be the length of the longest prefix with $d_j = g_{j-1}$
   for all $j \le n$. The round generates $d_1, \dots, d_n$ followed by $g_n$ (the
   correction at the first mismatch, or the bonus token when all $k'$ drafts were
   accepted): $n + 1$ tokens, never more than $r$.
4. **Stop or roll back.** If `max_new_tokens` is reached, stop. Otherwise bring the
   cache of **every** layer back to verified tokens only, and start the next round.

With $r = 1$ nothing is drafted: the round is one pass of all layers over the
unprocessed tokens.

- `prompt`: a list of $P \ge 1$ ints.
- `exit_layer`: an int $E$ with $1 \le E < L$.
- `k`: an int $\ge 1$; `max_new_tokens`: an int $\ge 1$.
- Returns the generated tokens (without the prompt) as a list of
  `max_new_tokens` Python ints.

**Cost limits.** Let $R$ be the number of rounds the procedure takes, $k'_i$ the
number of drafts in round $i$, and $T = \sum_i (k'_i + 1)$ the number of positions
scored.

| counter | limit | why |
|---|---|---|
| `layer_calls` | $E \cdot T + (L - E) \cdot R$ | per round, $k' + 1$ draft steps through $E$ layers each, then the $L - E$ late layers **once** |
| `layer_tokens` | $L \cdot (P - 1 + T)$ | every layer processes each prompt token once, and each scored position once |

Any loop that returns the right tokens within both limits is accepted. Running
the early layers again during verification, or keeping no cache between rounds,
exceeds the token limit. Verifying position by position, drafting tokens that can
no longer be emitted, dropping the bonus token, or drafting from an exit state
that was not normalized (which lowers the agreement and adds rounds) exceeds the
call limit.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| hidden states `h` | `(T, d_model)` for `T` positions | float32 |
| `lm.head(lm.norm(h))` | `(T, vocab_size)` | float32 |
| result | list of `max_new_tokens` ints | Python `int` |

## Example

A four-layer model whose last two layers mostly refine, exit after layer 2,
$P = 4$, $k = 4$:

```python
self_speculative_greedy(lm, [0, 17, 42, 5], 2, 4, 12)
# [32, 46, 15, 30, 9, 9, 9, 9, 53, 52, 24, 46]
```

| round | $r$ | drafts $d_1 \dots d_{k'}$ (exit) | full model $g_0 \dots$ | $n$ | generated |
|---|---|---|---|---|---|
| 1 | 12 | `32 46 15 34` | `32 46 15 30` ... | 3 | `32 46 15 30` |
| 2 | 8 | `9 53 22 52` | `9 9` ... | 1 | `9 9` |
| 3 | 6 | `9 53 47 50` | `9 9` ... | 1 | `9 9` |
| 4 | 4 | `53 52 24` | `53 52 24 46` | 3 | `53 52 24 46` |

In round 1 the exit agrees with the full model three times; at the fourth position
it says 34 and the full model says 30. Four tokens are generated, and the cache of
every layer, which covers the prompt and four drafts, is rolled back to the seven
verified tokens that have been processed (the correction 30 has not). In rounds 2
and 3 the exit expects 53 after a 9 where the full model repeats the 9. In round
4 only four tokens are left, so three are drafted; all are accepted and the bonus
token completes the output.

What the reference feeds, per round:

| round | early layers, one call each (positions) | late layers, one call | layer calls | layer-tokens |
|---|---|---|---|---|
| 1 | `0 17 42 5`, `32`, `46`, `15`, `34` | 8 positions | $5 \cdot 2 + 2$ | $8 \cdot 4$ |
| 2 | `30`, `9`, `53`, `22`, `52` | 5 positions | $5 \cdot 2 + 2$ | $5 \cdot 4$ |
| 3 | `9`, `9`, `53`, `47`, `50` | 5 positions | $5 \cdot 2 + 2$ | $5 \cdot 4$ |
| 4 | `9`, `53`, `52`, `24` | 4 positions | $4 \cdot 2 + 2$ | $4 \cdot 4$ |

$R = 4$ and $T = 5 + 5 + 5 + 4 = 19$: 46 layer calls (limit
$2 \cdot 19 + 2 \cdot 4 = 46$) and 88 layer-tokens (limit $4 \cdot (3 + 19) = 88$).
Plain cached greedy decoding takes $12 \cdot 4 = 48$ layer calls for the same
tokens: with this many rejections self-speculation barely pays. On a 40-token run
of a model whose exit agrees more often, the same loop takes 100 layer calls where
greedy decoding takes 160.

## Constraints

- $2 \le L \le 6$, $1 \le E < L$, $1 \le k \le 8$, $1 \le P$, and
  $P +$ `max_new_tokens` $+\, k \le$ `lm.max_context` (128).
- Tokens are compared exactly with the full model's greedy decode. Wherever an
  argmax is used, at the exit or after the last layer, its two largest logits
  differ by at least `1e-3`, so the way calls are split (which changes logits by
  about `1e-5`) affects neither the tokens nor the rounds.
- The two counters are compared with the limits above; $R$ and $T$ are computed
  by running the round procedure. A solution over a limit fails even when its
  tokens are right.
- Ties in an argmax go to the lower token id (this does not occur in the tests).

## Why this matters

A draft model is a second model: more memory, a second KV cache, and a draft that
shares nothing with the target's computation. Self-speculation removes all three.
The draft is the first layers of the model itself, so the cache is shared, and
verification does not start over: it continues the same forward pass from the
exit layer, on states that were saved while drafting. That reuse is the whole
economy of the method, and it is easy to give away. Run the full model on the
drafts "to verify them" and each position pays for the early layers twice; the
output is still right and the speedup is gone. Roll back the early layers and
forget the late ones, and the output is fluent and wrong. Apply the output head to
the raw exit state, and the model stays correct while its drafts get worse, a
slowdown that looks like a model-quality problem and is a missing norm.
