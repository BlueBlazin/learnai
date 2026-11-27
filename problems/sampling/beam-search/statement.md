Implement batched beam search over a transformer language model with a KV cache. At each step, keep the `n_beams` most probable partial hypotheses, extend all of them in a single batched model call, and return the best finished hypotheses.

## Specification

```python
def beam_search(lm, prompt, n_beams, max_new_tokens, eos_id, length_penalty): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) with vocabulary size `V`. For a prompt of length `P` or a batch of `B` live beams, you will use three operations:

```python
logits, cache = lm(tokens)             # tokens: list of P ints -> logits (P, V)
logits, cache = lm(tokens, cache)      # tokens: (B, 1) nested list -> logits (B, 1, V)
cache = cache.select(rows)             # a cache whose batch rows are `rows` of this one
```

`cache.select(rows)` takes a list of integer row indices, which may repeat or omit rows. For example, `cache.select([0, 0, 0])` expands the single-row prompt cache into three identical rows, and `cache.select([2, 0, 0])` sets new row 0 to old row 2 and new rows 1 and 2 to old row 0. When passing a cache to `lm`, `tokens` must have the same number of rows as `cache`. Caches are immutable, and `lm.counters.forward_calls` and `lm.counters.tokens_processed` track model work.

**Hypotheses and scores.** A hypothesis is a sequence of generated tokens $y_1, \ldots, y_L$ (excluding the prompt). Its *log-probability* is

$$
S = \sum_{t=1}^{L} \log p(y_t \mid \text{prompt}, y_1, \ldots, y_{t-1}),
$$

where $\log p$ is the log-softmax of the model's next-token logits. A hypothesis is *finished* when its last token is `eos_id` or when it reaches `max_new_tokens` tokens. With $\alpha$ = `length_penalty`, the *score* of a finished hypothesis divides $S$ by a power of its generated length $L$ (counting the final EOS token if present):

$$
\text{score} = \frac{S}{L^{\alpha}} .
$$

**The search.** Maintain an ordered list of *live* beams (unfinished hypotheses paired with their cumulative log-probabilities $S$) and a collection of finished hypotheses. Start with one live beam: the empty hypothesis with $S = 0$. Each step $t = 1, 2, \ldots$ runs five substeps:

1. **Score candidates.** Every live beam at index $i$ (with log-probability $S_i$) and every vocabulary token $v$ form a candidate extending beam $i$ by $v$ with log-probability $S_i + \log p(v \mid \ldots)$. Rank all candidates by log-probability descending, breaking ties by lower $i$, then lower $v$. Keep the top $2 \cdot \texttt{n\_beams}$ candidates. Their positions $0, 1, 2, \ldots$ are their *ranks*.
2. **Partition into finished and live.** Walk those candidates in rank order.
   - If $t < \texttt{max\_new\_tokens}$: a candidate whose new token is `eos_id` becomes finished **only if its rank is below `n_beams`** (otherwise discard it). Any other candidate becomes a live beam for the next step until `n_beams` live beams have been collected, preserving their rank order.
   - If $t = \texttt{max\_new\_tokens}$: every candidate with rank below `n_beams` becomes finished regardless of its last token, and there are no live beams.
3. **Prune finished hypotheses.** Keep at most `n_beams` finished hypotheses. If there are more, drop the ones with the lowest scores (for equal scores, drop the one added later).
4. **Check the stop condition.** Stop after this step if $t = \texttt{max\_new\_tokens}$, or if the finished set contains `n_beams` hypotheses and no live beam can beat the worst finished score:

   $$
   \frac{S_{\text{best}}}{\texttt{max\_new\_tokens}^{\alpha}} \;\le\;
   \text{the lowest score in the finished set},
   $$

   where $S_{\text{best}}$ is the largest log-probability among the new live beams. Because log-probabilities are non-positive and no hypothesis exceeds `max_new_tokens` tokens, the left side upper-bounds the final score of any future extension of a live beam.
5. **Advance.** Extend all live beams by one token in a single batched model call.

Keeping $2 \cdot \texttt{n\_beams}$ candidates guarantees `n_beams` live beams remain even if every beam's top continuation is EOS. Restricting finishes to ranks below `n_beams` ensures a hypothesis can only finish if it would have survived beam pruning.

**Parameters and return value.**

- `prompt`: a list of $P \ge 1$ token IDs.
- `n_beams`: an int with $1 \le \texttt{n\_beams} \le 8$.
- `max_new_tokens`: an int $\ge 1$.
- `eos_id`: an int, or `None` if no token ends a hypothesis early.
- `length_penalty`: a float $\alpha \ge 0$. At $\alpha = 0$ the score is the raw log-probability (favoring shorter hypotheses), and larger values favor longer ones.
- Returns the finished hypotheses as a list of exactly `n_beams` tuples `(tokens, score)`, sorted **best score first** (for equal scores, the one added earlier comes first). Each `tokens` is a list of Python ints (including the final EOS if present), and `score` is a Python float.

**Cost limits.** For a search that stops after step $s$, both limits are exact:

| counter | limit | why |
|---|---|---|
| `forward_calls` | $s$ | one call for the prompt (producing step 1 candidates), then one batched call before each subsequent step |
| `tokens_processed` | $P + \texttt{n\_beams} \cdot (s - 1)$ | the prompt once as a single row, then one token per live beam on each later call |

You must not repeat the prompt per beam, re-feed beam prefixes, or make a model call after the stop condition in substep 4 holds.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(P, V)` for the prompt; `(B, 1, V)` for a batched call | float32 |
| result | list of `n_beams` tuples `(list of ints, float)` | Python |

## Example

```python
lm = TinyLM(fw, seed=6)          # vocab_size 64

beam_search(lm, [0, 63, 33, 13], n_beams=2, max_new_tokens=12, eos_id=52, length_penalty=0.0)
# [([52], -1.3500), ([38, 4, 52], -3.6046)]          3 calls, 4 + 2 * 2 = 8 tokens
```

The top four candidates ($2 \cdot \texttt{n\_beams}$) at each step, with their log-probabilities:

| step | live beams before | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|---|
| 1 | `[]` | `[52]` −1.350, **finished** | `[38]` −2.216, live | `[30]` −2.323, live | `[34]` −3.234 |
| 2 | `[38]`, `[30]` | `[38, 4]` −3.412, live | `[30, 38]` −3.499, live | `[30, 61]` −3.943 | `[38, 43]` −4.303 |
| 3 | `[38, 4]`, `[30, 38]` | `[38, 4, 52]` −3.605, **finished** | `[30, 38, 4]` −4.790, live | `[30, 38, 55]` −4.980, live | `[30, 38, 34]` −6.205 |

In step 1, the highest-scoring token is EOS (`52`). It finishes at rank 0, and ranks 1 and 2 become the live beams. The prompt cache has one row, so the call before step 2 passes `[[38], [30]]` with `cache.select([0, 0])`. In step 3, both new live beams extend old row 1, so the next call would use `cache.select([1, 1])`. However, that call is never made because the finished set is full and the best live beam has $S = -4.790$, which is below the worst finished score of $-3.605$. With $\alpha = 0$ the upper bound equals $S$, so the search stops after 3 of 12 steps.

![The three steps of this example and its stop. Arrows run from a live beam, one cache row each, to the candidates that extend it.](assets/beam-steps.svg)

A length penalty changes both hypothesis ranking and when the search stops:

```python
lm = TinyLM(fw, seed=10)
beam_search(lm, [0, 43, 22, 24, 46], n_beams=2, max_new_tokens=5, eos_id=36, length_penalty=1.0)
# [([5, 36], -1.3937), ([35, 61, 39, 29, 58], -1.4393)]          5 calls, 5 + 2 * 4 = 13 tokens
```

After step 3, the finished set holds `[5, 36]` ($-2.787 / 2 = -1.394$) and `[5, 5, 36]` ($-5.136 / 3 = -1.712$), while the best live beam has $S = -5.511$. Its upper bound is $-5.511 / 5 = -1.102$, which exceeds $-1.712$, so the search continues. At step 5, `[35, 61, 39, 29, 58]` finishes with $-7.197 / 5 = -1.439$ and displaces `[5, 5, 36]`.

## Constraints

- $1 \le P$, $P + \texttt{max\_new\_tokens} \le \texttt{lm.max\_context}$ (128), and the vocabulary size is 64, so there are always at least $2 \cdot \texttt{n\_beams}$ candidates.
- Returned token lists and their order are compared exactly against an uncached, unbatched beam search. Scores are checked to relative tolerance `1e-4` and absolute tolerance `1e-5`.
- Logits from a cached, batched call differ from an uncached call by about `1e-5`. The test suite keeps every comparison (adjacent candidates, finished scores, and the stopping bound) at least `1e-3` apart, so tie rules never decide a test and rounding cannot flip the outcome.
- Model call and token counters must stay within the exact limits above. Exceeding either limit fails even if the returned hypotheses are correct.
- You may use your framework's `log_softmax`, sort, and top-k operations, or `ops.log_softmax` from the kit. A Python loop over the $2 \cdot \texttt{n\_beams}$ candidates is fine, but do not loop over the vocabulary in Python.

## Why this matters

Beam search remains the standard decoder for speech recognition (such as Whisper), machine translation, and constrained or reranked generation pipelines. Its batched execution pattern, where multiple hypotheses share a single forward pass, also underlies parallel sampling (`n > 1`), speculative decoding, and tree search.

In practice, cache reordering is the most dangerous failure mode. If beams are reshuffled without gathering the matching cache rows, each row silently continues another beam's history and produces fluent but wrong text without raising an error. Scoring details matter just as much. Normalizing live scores too early, counting the prompt in the length, or stopping as soon as `n_beams` hypotheses finish all change the output, which is why different libraries often return different beam search results for the same model.
