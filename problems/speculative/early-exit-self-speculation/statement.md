Implement self-speculative greedy decoding, where a single model drafts tokens for itself. The first layers and shared output head propose draft tokens, and the remaining layers verify them in one call without recomputing any early-layer positions. The output must match the full model's greedy decode token for token within strict layer-call and layer-token limits.

![This problem is Draft and the target call, here the lower and upper layers of one model.](assets/map.svg)

## Specification

```python
def self_speculative_greedy(lm, prompt, exit_layer, k, max_new_tokens): ...
```

`lm` is a [`LayeredLM`](../../../python/src/learnai/kit/layeredlm.py), a decoder transformer with $L$ = `lm.n_layers` layers whose stack can be run one layer range at a time.

```python
h = lm.embed(tokens, offset=n)  # (N, d_model): tokens at positions n, n + 1, ...
h, cache = lm.layers(h, a, b, cache)  # layers a .. b - 1 on those N positions
logits = lm.head(lm.norm(h))  # (N, vocab_size): output logits after the final norm
cache = lm.empty_cache()  # empty KV cache with 0 tokens in every layer
cache = cache.truncate(n)  # roll every layer back to its first n tokens
```

- `lm.layers(h, a, b, cache)` runs layers `a` through `b - 1` on hidden states `h` and returns the updated hidden states along with a new cache (caches are immutable). The rows of `h` represent the next unprocessed positions for those layers, in order, and layers `a` through `b - 1` must all have cached the same number of tokens. Early layers may be ahead of late layers. `cache.lengths` lists the number of cached tokens per layer, and `cache.length` is their maximum.
- The full model's logits for a sequence are `lm.head(lm.norm(h))` where `h` is the output of layer $L - 1$. Letting $E$ = `exit_layer`, an *early exit* after layer $E - 1$ applies the same `lm.norm` and `lm.head` to the output of layer $E - 1$. Because `lm.norm` has a learned gain, calling `lm.head(h)` without `lm.norm` computes a different function with a different argmax.
- `lm.counters` tracks two costs. `layer_calls` counts one per layer per call to `lm.layers`, regardless of how many tokens are in `h`. `layer_tokens` counts layers multiplied by the number of tokens in `h`. Calls to `embed`, `norm`, and `head` are free.

**Return value.** Return the full model's greedy continuation of `prompt` (without the prompt itself) as a list of `max_new_tokens` Python `int`s. Each token must be the argmax of the full model's next-token logits given the prompt and all previously generated tokens.

**Procedure.** Let $S$ be the verified sequence (the prompt followed by all tokens generated so far), and let $r$ be the number of tokens still left to generate. Each *round* runs four steps:

1. **Draft with layers $0 \dots E - 1$.** Propose $k' = \min(k,\, r - 1)$ draft tokens $d_1, \dots, d_{k'}$. Each draft $d_j$ is the argmax of the early exit's logits at the last position of $S, d_1, \dots, d_{j-1}$. The early layers run on every token they have not yet processed. In the first round that begins with the entire prompt, and in later rounds it begins with the newest token of $S$, followed by each draft one by one. Feed $d_{k'}$ through the early layers as well so that its exit state is available for verification.
2. **Verify with layers $E \dots L - 1$.** Make one call to `lm.layers(..., E, L, ...)` on the saved exit states of all positions the early layers just processed. Applying `lm.norm` and `lm.head` to the last $k' + 1$ rows gives the full model's greedy tokens $g_0, \dots, g_{k'}$, where $g_0$ follows $S$ and each $g_j$ follows $S, d_1, \dots, d_j$.
3. **Accept.** Let $n$ be the length of the longest prefix where $d_j = g_{j-1}$ for all $1 \le j \le n$. Append $d_1, \dots, d_n$ followed by $g_n$ (the correction at the first mismatch, or the bonus token if all $k'$ drafts match). This emits $n + 1$ tokens, never exceeding $r$.
4. **Stop or roll back.** If `max_new_tokens` tokens have been generated, stop and return them. Otherwise roll every layer's cache back to the verified tokens already processed, and start the next round.

![The first $E$ layers draft through the model's own norm and head, and the remaining layers verify from the same exit states.](assets/exit-split.svg)

When $r = 1$, $k' = 0$ and nothing is drafted. The round runs the early layers once on the unprocessed tokens and the late layers once on those exit states.

**Cost limits.** Let $P$ be the length of `prompt`, $R$ be the number of rounds taken by the procedure above, $k'_i$ be the number of drafts in round $i$, and $T = \sum_i (k'_i + 1)$ be the total number of positions scored.

| counter | limit | why |
|---|---|---|
| `layer_calls` | $E \cdot T + (L - E) \cdot R$ | per round, $k' + 1$ draft steps through $E$ layers each, then the $L - E$ late layers once |
| `layer_tokens` | $L \cdot (P - 1 + T)$ | every layer processes each prompt token once and each scored position once |

Any implementation that returns the exact greedy tokens within both limits is accepted. Re-running the early layers during verification or dropping the cache between rounds exceeds `layer_tokens`. Verifying one position at a time, drafting past $r - 1$, dropping the bonus token, or omitting `lm.norm` at the exit (which lowers agreement and adds rounds) exceeds `layer_calls`.

## Shapes

| name | shape / type | dtype |
|---|---|---|
| `prompt` | list of $P$ ints ($P \ge 1$) | Python `int` |
| `exit_layer` | scalar $E$ ($1 \le E < L$) | Python `int` |
| `k`, `max_new_tokens` | scalars ($\ge 1$) | Python `int` |
| hidden states `h` | `(N, d_model)` for `N` positions | float32 |
| `lm.head(lm.norm(h))` | `(N, vocab_size)` | float32 |
| return value | list of `max_new_tokens` ints | Python `int` |

## Example

Consider a four-layer model ($L = 4$) whose last two layers mostly refine earlier predictions, with `exit_layer = 2`, $P = 4$, and $k = 4$:

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

In round 1, the early exit agrees with the full model on the first three drafts, then proposes `34` where the full model chooses `30`. The round emits four tokens (`32, 46, 15, 30`). The cache covers the four prompt tokens plus four drafts, so every layer is truncated to the seven verified tokens already processed (`0, 17, 42, 5, 32, 46, 15`), leaving the correction `30` to be fed at the start of round 2. In rounds 2 and 3, the exit predicts `53` after `9` while the full model repeats `9`. In round 4, only four tokens remain ($r = 4$), so three are drafted. All three match, and the bonus token `46` finishes the output.

Each round feeds the following positions:

| round | early layers, one call each (positions) | late layers, one call | layer calls | layer-tokens |
|---|---|---|---|---|
| 1 | `0 17 42 5`, `32`, `46`, `15`, `34` | 8 positions | $5 \cdot 2 + 2$ | $8 \cdot 4$ |
| 2 | `30`, `9`, `53`, `22`, `52` | 5 positions | $5 \cdot 2 + 2$ | $5 \cdot 4$ |
| 3 | `9`, `9`, `53`, `47`, `50` | 5 positions | $5 \cdot 2 + 2$ | $5 \cdot 4$ |
| 4 | `9`, `53`, `52`, `24` | 4 positions | $4 \cdot 2 + 2$ | $4 \cdot 4$ |

Across $R = 4$ rounds and $T = 5 + 5 + 5 + 4 = 19$ scored positions, this uses 46 layer calls (limit $2 \cdot 19 + 2 \cdot 4 = 46$) and 88 layer-tokens (limit $4 \cdot (3 + 19) = 88$). Plain cached greedy decoding takes $12 \cdot 4 = 48$ layer calls for the same 12 tokens, so with this many rejections the savings are small. On a 40-token run where the exit agrees more often, the same loop takes 100 layer calls compared to 160 for greedy decoding.

## Constraints

- $2 \le L \le 6$, $1 \le E < L$, $1 \le k \le 8$, $1 \le P$, and $P + \text{max\_new\_tokens} + k \le \text{lm.max\_context}$ (128).
- Generated tokens are compared exactly against the full model's greedy decode. At every argmax (both early exit and full model), the top two logits differ by at least `1e-3`, so small floating-point differences from splitting calls (around `1e-5`) do not change the tokens or the rounds.
- Both counters are checked against the limits above, where $R$ and $T$ come from running the round procedure. Exceeding either limit fails even if the tokens are correct.
- Argmax ties go to the lower token id (ties do not occur in the tests).

## Why this matters

Standard speculative decoding requires a separate draft model, which adds weight memory, a second KV cache, and draft work that the target model throws away. Self-speculation removes all three costs. Because the first layers of the target act as the drafter, both phases share one KV cache and verification continues the same forward pass from the saved exit states.

That reuse is easy to break silently. Running the full model on the drafts during verification still produces exact tokens, but pays for the early layers twice and erases the speedup. Truncating only the early layers' cache leaves rejected tokens in the late layers, producing fluent but wrong output. Calling `lm.head` without `lm.norm` at the exit keeps the final output correct while sharply reducing the acceptance rate, which looks like a model-quality issue when it is actually a missing normalization call.
