## The idea

A speculative draft only needs to be cheap and frequently right. When generation repeats material already in the context—such as quoting a document, editing code, or continuing a structured list—the context itself is a strong predictor. Searching for an earlier occurrence of the current tail and copying what followed replaces a draft-model forward pass with a fast sequence search. The target model's verification step guarantees exact output no matter how often the guess misses.

Three rules govern the search:

- **Longest n-gram first.** Matching three tokens is much stronger evidence of a true repeat than matching one. Shorter n-grams serve as a fallback, and `min_ngram` sets the floor below which a match is too weak to justify verification overhead.
- **Exclude only the suffix itself.** The suffix always matches itself at index $L - n$, where no tokens follow. Restricting starts to $s < L - n$ avoids that trivial match while still allowing windows that overlap the suffix. Overlap is what makes runs (`7 7 7 7`) and short cycles (`1 2 1 2 1 2`) draftable.
- **Earliest occurrence first.** When an n-gram appears multiple times, taking the smallest valid $s$ matches the convention used by major serving frameworks. The earliest occurrence also has the most tokens after it, so the draft is less likely to be truncated by the end of `context`.

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

The outer bound `min(max_ngram, length - 1)` handles short contexts without a separate branch, because the range is empty whenever $L \le 1$. The inner loop scans legal start indices from left to right and stops before `length - n`, so the first hit is guaranteed to be the earliest non-trivial occurrence. Slicing `context[start + n : start + n + k]` returns a new list and automatically clamps at the end of `context`.

## Common mistakes

- **Taking the most recent occurrence.** Scanning right to left picks the latest match instead of the earliest. That is a reasonable heuristic in general, but it disagrees with this specification whenever an n-gram appears multiple times with different continuations.
- **Matching the suffix with itself.** Allowing `start` to reach $L - n$ creates a trivial match for every n-gram that has no earlier occurrence. Because nothing follows index $L - n$, the function returns `[]` instead of falling back to a shorter n-gram. Nothing crashes, but speculation silently stops firing.
- **Shortest n-gram first.** Looping $n$ upward anchors the copy on a single common token (like a comma or space) instead of a longer phrase that uniquely identifies the right location.
- **Forbidding overlap.** Requiring $s + n \le L - n$ looks tidy, but it breaks on repetitive runs and short cycles where prompt lookup is most effective.
- **Ignoring `min_ngram`.** Falling through to $n = 1$ drafts from single-token coincidences. Those drafts are mostly rejected, and every rejected draft still makes the target forward pass longer.
- **Copying from the wrong offset.** The continuation starts at $s + n$. Starting at $s + n - 1$ repeats the last context token as the first draft token, which causes greedy verification to reject the entire draft immediately.
- **Insisting on `k` tokens.** Skipping an occurrence because fewer than `k` tokens follow it changes which match is chosen. This specification returns the shorter draft from the earliest match (unlike the original prototype described below).

## Complexity

Let $N =$ `max_ngram`. For each of at most $N$ lengths, the direct search compares $O(L)$ windows of length at most $N$. This takes $O(L N^2)$ worst-case time per call and $O(N + k)$ extra memory.

Called once per decoding step, a direct search is quadratic in total sequence length across a full generation. Production engines reduce this to $O(L)$ per call using the KMP failure function on the reversed sequence (as in vLLM), or to amortized $O(1)$ per token using an incrementally updated n-gram hash table or suffix automaton.

## In the wild

Apoorv Saxena introduced prompt lookup decoding in November 2023 (`github.com/apoorvumang/prompt-lookup-decoding`) with defaults `max_ngram_size = 3` and `num_pred_tokens = 10`, searching n-gram lengths down to 1. Hugging Face `transformers` adopted it as `generate(..., prompt_lookup_num_tokens=10)`, and vLLM implements it as the `ngram` speculative method with parameters `prompt_lookup_max`, `prompt_lookup_min`, and `num_speculative_tokens`—the exact three parameters of this problem.

All three implementations share the core rule used here: try the longest n-gram first, and pick the **earliest** occurrence among ties. The original code and `transformers` scan left to right and stop at the first usable match (a comment in `transformers` notes that other policies could be explored). vLLM's proposer runs the KMP failure function over the reversed sequence and explicitly retains the earliest position among the longest matches. They differ only on edge cases:

- **Short continuations.** The original prototype accepts a match only if a full `num_pred_tokens` follow it and the continuation starts before the suffix begins (forbidding overlapping or adjacent matches). Otherwise it tries the next match, then a shorter n-gram. Both `transformers` and vLLM instead exclude only the suffix itself and clamp the copy at the end of the sequence, which allows overlapping matches. This problem follows `transformers` and vLLM.
- **Minimum length.** The original code and `transformers` search down to $n = 1$, while vLLM stops at `prompt_lookup_min` (our `min_ngram`).
- **Extra limits.** `transformers` also truncates drafts before an EOS token or the generation length limit, and vLLM clamps `k` to the model's maximum context length.

Regardless of these variations, the target model verifies the drafts identically to neural drafts. Under greedy decoding, the generated tokens remain identical to standard decoding, and only the acceptance rate changes.
