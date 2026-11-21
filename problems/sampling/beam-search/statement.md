Implement beam search over a transformer language model: keep the `n_beams` most
probable partial outputs, extend all of them with **one batched model call per
step**, and return the best finished hypotheses. Three things make it harder than
greedy decoding: the beams are reshuffled at every step, so the KV cache has to be
reordered with them; hypotheses finish at different lengths and must be compared
fairly; and the search should stop as soon as nothing better can still be found.

## Specification

```python
def beam_search(lm, prompt, n_beams, max_new_tokens, eos_id, length_penalty): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py). What you need
from it:

```python
logits, cache = lm(tokens)             # tokens: list of P ints -> logits (P, V)
logits, cache = lm(tokens, cache)      # tokens: (B, 1) nested list -> logits (B, 1, V)
cache = cache.select(rows)             # a cache whose batch rows are `rows` of this one
```

`cache.select(rows)` takes a list of row indices; rows may repeat and may be left
out. `cache.select([0, 0, 0])` turns the one-row cache of the prompt into three
identical rows, and `cache.select([2, 0, 0])` makes new row 0 a copy of old row 2 and
new rows 1 and 2 copies of old row 0. A call with a cache must pass as many rows of
tokens as the cache has. Caches are immutable; `lm.counters.forward_calls` and
`lm.counters.tokens_processed` count the work.

**Hypotheses and scores.** A hypothesis is a sequence of generated tokens
$y_1, \ldots, y_L$ (the prompt is not part of it). Its *log-probability* is

$$
S = \sum_{t=1}^{L} \log p(y_t \mid \text{prompt}, y_1, \ldots, y_{t-1}),
$$

where $\log p$ is the log-softmax of the model's next-token logits. A hypothesis is
*finished* when its last token is `eos_id` or when it has `max_new_tokens` tokens.
The *score* of a finished hypothesis divides by a power of its length, with
$\alpha$ = `length_penalty`:

$$
\text{score} = \frac{S}{L^{\alpha}} ,
$$

where $L$ counts every generated token, the EOS included.

**The search.** It keeps a list of *live* beams (unfinished hypotheses with their
$S$), in order, and a set of finished hypotheses. Initially the only live beam is the
empty hypothesis with $S = 0$. Step $t = 1, 2, \ldots$ does the following.

1. **Candidates.** Every live beam $i$ (its position in the list) and every token
   $v$ form a candidate, the beam extended by $v$, with log-probability
   $S_i + \log p(v \mid \ldots)$. Rank all candidates by log-probability, highest
   first; equal values are ranked by lower $i$, then lower $v$. Take the first
   $2 \cdot$ `n_beams` of them; their positions $0, 1, 2, \ldots$ are their *ranks*.
2. **Sort them into finished and live**, going through the ranks in order. If
   $t <$ `max_new_tokens`:
   - a candidate whose new token is `eos_id` is finished **if its rank is below
     `n_beams`**; with a rank of `n_beams` or more it is discarded;
   - any other candidate becomes a live beam of the next step, until `n_beams` live
     beams have been collected. The live beams keep the order of their ranks.

   If $t =$ `max_new_tokens`, the candidates of rank below `n_beams` are finished,
   whatever their last token, and there are no live beams.
3. **Keep the best.** The finished set holds at most `n_beams` hypotheses: when it
   has more, the ones with the lowest scores are removed (of two equal scores, the
   one that was added later).
4. **Stop?** The search ends after this step if $t =$ `max_new_tokens`, or if the
   finished set holds `n_beams` hypotheses and no live beam can still beat the worst
   of them:

   $$
   \frac{S_{\text{best}}}{\texttt{max\_new\_tokens}^{\alpha}} \;\le\;
   \text{the lowest score in the finished set},
   $$

   where $S_{\text{best}}$ is the largest log-probability among the new live beams.
   The left side is an upper bound on the score of anything a live beam can become:
   log-probabilities are negative and only decrease as tokens are added, and no
   hypothesis is longer than `max_new_tokens`.
5. **Otherwise advance** all live beams by one token with a single model call.

Taking $2 \cdot$ `n_beams` candidates guarantees that `n_beams` live beams remain
even if every beam's best continuation is EOS. The rank condition in rule 2 means
that a hypothesis may only finish if it would have survived pruning to `n_beams`.

**Parameters and result.**

- `prompt`: a list of $P \ge 1$ token ids.
- `n_beams`: an int, $1 \le$ `n_beams` $\le 8$.
- `max_new_tokens`: an int $\ge 1$.
- `eos_id`: an int or `None` (no token finishes a hypothesis early).
- `length_penalty`: a float $\alpha \ge 0$. With 0 the score is the log-probability,
  which favors short hypotheses; larger values favor longer ones.
- Returns the finished set as a list of exactly `n_beams` tuples `(tokens, score)`,
  **best score first** (equal scores: the one added earlier first). `tokens` is a
  list of Python ints, including the final EOS if there is one; `score` is a Python
  float.

**Cost limits.** For a search that ends after step $s$:

| counter | limit | why |
|---|---|---|
| `forward_calls` | $s$ | the prompt (which gives the candidates of step 1), then one batched call before each later step |
| `tokens_processed` | $P + \texttt{n\_beams} \cdot (s - 1)$ | the prompt once, in one row; then one token per live beam and call |

So the prompt is not repeated per beam, no beam's prefix is fed again, and no call
is made after the stop condition of rule 4 holds. Both limits are exact.

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

The four best candidates of each step (2 · `n_beams`), with their log-probabilities:

| step | live beams before | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|---|
| 1 | `[]` | `[52]` −1.350, **finished** | `[38]` −2.216, live | `[30]` −2.323, live | `[34]` −3.234 |
| 2 | `[38]`, `[30]` | `[38, 4]` −3.412, live | `[30, 38]` −3.499, live | `[30, 61]` −3.943 | `[38, 43]` −4.303 |
| 3 | `[38, 4]`, `[30, 38]` | `[38, 4, 52]` −3.605, **finished** | `[30, 38, 4]` −4.790, live | `[30, 38, 55]` −4.980, live | `[30, 38, 34]` −6.205 |

In step 1 the most probable token is EOS. It finishes at rank 0, and the next two
candidates become the live beams. The cache of the prompt has one row; the call
before step 2 feeds `[[38], [30]]` with `cache.select([0, 0])`. In step 3 both new
live beams extend old row 1, so the next call would use `cache.select([1, 1])`. It
is never made: the finished set is now full, and the best live beam has
$S = -4.790$, which is below the worst finished score, $-3.605$. With
$\alpha = 0$ the bound is $S$ itself, so the search stops after 3 of 12 steps.

![The three steps of this example and its stop. Arrows run from a live beam, one cache row each, to the candidates that extend it.](assets/beam-steps.svg)

A length penalty changes both the ranking of the results and the stopping point:

```python
lm = TinyLM(fw, seed=10)
beam_search(lm, [0, 43, 22, 24, 46], n_beams=2, max_new_tokens=5, eos_id=36, length_penalty=1.0)
# [([5, 36], -1.3937), ([35, 61, 39, 29, 58], -1.4393)]          5 calls, 5 + 2 * 4 = 13 tokens
```

After step 3 the finished set is `[5, 36]` ($-2.787 / 2 = -1.394$) and `[5, 5, 36]`
($-5.136 / 3 = -1.712$), and the best live beam has $S = -5.511$. The bound is
$-5.511 / 5 = -1.102$, above $-1.712$: a live beam may still win, so the search goes
on. At step 5 the hypothesis `[35, 61, 39, 29, 58]` finishes with
$-7.197 / 5 = -1.439$ and displaces `[5, 5, 36]`.

## Constraints

- $1 \le P$, $P +$ `max_new_tokens` $\le$ `lm.max_context` (128), and the vocabulary
  has 64 tokens, so there are always at least $2 \cdot$ `n_beams` candidates.
- The token lists and their order are compared exactly with a beam search that uses
  no cache and no batching. Scores are compared to a relative tolerance of `1e-4`
  and an absolute tolerance of `1e-5`.
- Logits from a cached, batched call differ from an uncached call by about `1e-5`.
  The tests keep every comparison the search makes (between neighboring candidates
  in the ranking, between finished scores, and in the stopping rule) at least `1e-3`
  apart, so the tie rules above never decide a test and rounding cannot change the
  result.
- The counters are compared with the limits above. A solution over a limit fails
  even when its hypotheses are right.
- You may use your framework's `log_softmax`, sort, and top-k, or `ops.log_softmax`
  from the kit. A Python loop over the $2 \cdot$ `n_beams` candidates is fine; do not
  loop over the vocabulary.

## Why this matters

Beam search is still the decoder of speech recognition (Whisper), translation, and
every constrained or reranked generation pipeline, and its structure, many
hypotheses sharing one batched forward pass, is the same one that parallel sampling
(`n > 1`), speculative decoding, and tree search use. The cache gather is where it
goes wrong in practice: when the beams are reordered and the cache is not, every
row silently continues another row's history. The output is fluent and wrong, and
nothing crashes. The scoring rules matter just as much: normalizing too early,
counting the prompt in the length, or stopping at the first finished hypotheses all
change which text comes out, which is why "beam search" gives different results in
different libraries.
