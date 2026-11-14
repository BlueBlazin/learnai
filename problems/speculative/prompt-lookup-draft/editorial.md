## The idea

A draft only has to be cheap and often right. When the output repeats material
that is already in the context (quoting a document, rewriting code with small
edits, continuing a list), the best predictor of the next tokens is the context
itself: find where the current tail occurred before, and guess that the same tokens
follow again. That costs a string search instead of a model call, and the target
model's verification step keeps the output exact however bad the guess is.

Three choices define the search, and each is a design decision with a reason:

- **Longest n-gram first.** A match on three tokens is much stronger evidence of a
  repeat than a match on one. Shorter n-grams are a fallback, and `min_ngram`
  decides when a match is too flimsy to be worth a wasted verification.
- **Not the suffix itself.** The suffix trivially equals itself, and nothing follows
  it. Only starts $s < L - n$ qualify. That bound still admits windows that overlap
  the suffix, which is what makes runs (`7 7 7 7`) and short cycles
  (`1 2 1 2 1 2`) draftable.
- **The earliest occurrence.** When the n-gram occurs several times, the first one
  in the context is used, as in the implementations this problem follows. The
  earliest occurrence is the one with the most text after it, so the draft is
  rarely cut short by the end of the context.

## Reference solution (Python)

```python
def lookup_draft(context, max_ngram, min_ngram, k):
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
```

The outer bound `min(max_ngram, length - 1)` handles every short context: for
$L \le 1$ the range is empty. The inner loop runs over the legal starts from the
left, so the first hit is the earliest occurrence, and `range(length - n)` ends just
before the suffix's own start. The slice both copies (the caller gets a new list)
and truncates at the end of the context.

## Common mistakes

- **Taking the most recent occurrence.** Scanning right to left selects the latest
  occurrence. It is a defensible policy (recent text may predict best), but it is
  not what the reference implementations do, and the two disagree whenever an
  n-gram occurs twice with different continuations.
- **Matching the suffix with itself.** Letting the start index reach $L - n$ finds
  a "match" for every n-gram that has no earlier occurrence: the suffix itself,
  followed by nothing. The function then returns an empty draft instead of trying a
  shorter n-gram, and nothing fails; speculation is just silently less effective.
- **Shortest n-gram first.** Looping $n$ upward anchors the copy on a single common
  token (a space, a comma) instead of the three-token phrase that identifies the
  place.
- **Forbidding overlap.** Requiring the occurrence to end before the suffix starts
  looks tidy and loses exactly the highly repetitive inputs where lookup is most
  effective.
- **Ignoring `min_ngram`.** Falling through to $n = 1$ produces drafts from
  one-token coincidences. They are mostly rejected, and each rejection has still
  cost the target a longer forward pass.
- **Copying from the wrong offset.** The continuation starts at $s + n$. Starting at
  $s + n - 1$ makes the first draft token a repeat of the last context token, and
  greedy verification stops at the first wrong token: the whole draft is lost.
- **Insisting on `k` tokens.** Skipping an occurrence because fewer than `k` tokens
  follow it changes which occurrence is used. A short draft from the specified
  place is the answer here (the original implementation chose otherwise; see below).

## Complexity

With $N =$ `max_ngram`, the direct search compares $O(L)$ windows of length at most
$N$ for each of at most $N$ lengths: $O(L N^2)$ time in the worst case and $O(N + k)$
extra memory. Called once per decoding round, that is quadratic in the sequence
length over a whole generation. Production implementations get this down to $O(L)$
per call with the KMP failure function on the reversed sequence (vLLM), or to
amortized $O(1)$ per token with an incrementally maintained hash from n-grams to
positions or with a suffix automaton.

## In the wild

Prompt lookup decoding was introduced by Apoorv Saxena in November 2023
(`github.com/apoorvumang/prompt-lookup-decoding`), with `max_ngram_size = 3` and
`num_pred_tokens = 10` as defaults and the n-gram length going from the maximum down
to 1. It was merged into Hugging Face `transformers` as
`generate(..., prompt_lookup_num_tokens=10)` and is the `ngram` speculative method
of vLLM, configured by `prompt_lookup_max`, `prompt_lookup_min`, and
`num_speculative_tokens`: the three parameters of this problem.

The three implementations agree on the core rule, which is the one specified here:
longest n-gram first, and among several occurrences the **earliest** one. The
original code and `transformers` scan matches left to right and stop at the first
usable one (a comment in `transformers` notes that another choice might be better),
and vLLM's proposer, which runs the KMP failure function over the reversed
sequence, explicitly keeps the earliest position among the longest matches. They
differ at the edges:

- **Short continuations.** The original accepts a match only if a full
  `num_pred_tokens` follow it and the continuation starts before the suffix begins
  (so the match cannot overlap or even touch the suffix); otherwise it moves on to
  the next match, then to a shorter n-gram. `transformers` and vLLM clamp the copy
  at the end of the sequence instead and exclude only the suffix itself, so
  overlapping matches count. This problem follows the latter two.
- **Minimum length.** The original and `transformers` shorten the n-gram down to 1;
  vLLM stops at `prompt_lookup_min`, the `min_ngram` of this problem.
- **Extras.** `transformers` also cuts the draft before an EOS token and at the
  generation length limit, and vLLM clamps `k` to the model's context length.

Whatever the rule, the drafts are verified exactly like a draft model's tokens, so
with greedy decoding the output is identical to the target's own; only the
acceptance rate depends on these choices.
