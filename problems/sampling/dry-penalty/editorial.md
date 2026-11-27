## The idea

DRY asks one question at each decoding step: *if the model emits token $x$ next, how
long is the verbatim repeat it would extend?* Wherever the window's trailing tokens
appeared earlier in the window, the token immediately following that earlier
occurrence is the one that would continue the repeat. The length of the matched run
measures how deep into a loop the model already is. For a match of length $M$,
repeats shorter than `allowed_length` (such as names, idioms, or list markers)
receive no penalty. At `allowed_length` or above, the penalty
$\text{multiplier} \cdot \text{base}^{M - \text{allowed\_length}}$ starts mild and
multiplies by `base` with each additional repeated token so no loop can survive long.

Two rules make this practical on real text:

- **Sequence breakers.** Chat templates, newlines, and quotation marks repeat
  legitimately and separate units of text. A match cannot contain a breaker, and a
  breaker is never penalized. Because every match is a suffix of the window, "cannot
  contain a breaker" is equivalent to capping every match length at the number of
  non-breaker tokens at the end of the window. That cap, `limit`, is computed once in
  a single scan from the back.
- **Longest match per token.** If a candidate token follows multiple earlier
  occurrences of the current suffix, only the longest match determines its penalty.

**Linear-time matching.** For each position $i$, the match length $m_i$ is the
longest common suffix of the prefix $w_0 \ldots w_i$ and the full window
$w_0 \ldots w_{n-1}$. Reversing the window into `rev` turns that into the longest
common prefix of `rev` and `rev[k:]`, where $k = n - 1 - i$. That array of longest
common prefix lengths is the **Z-array**.

The Z-algorithm computes the entire Z-array in $O(n)$ time by tracking the rightmost
matched interval `[left, right)` that equals a prefix of `rev`. When visiting a
position `k` inside `[left, right)`, the substring starting at `k` matches the
substring starting at `k - left`, whose Z-value is already known. If `z[k - left]`
stays strictly inside the remaining interval (`z[k - left] < right - k`), then `z[k]`
equals `z[k - left]` with no new character comparisons. Otherwise, at least
`right - k` tokens already match, and direct comparisons resume from `right`. Every
successful comparison advances `right`, which never decreases, so the total work is
linear.

By contrast, a naive scan compares backward from every earlier occurrence of the last
token. That is fast on ordinary text where most comparisons fail immediately, but on
a loop of period $p$, every $p$-th position matches all the way back to the start of
the loop, costing $O(n^2 / p)$ comparisons.

## Reference solution (Python)

```python
import math

FLOAT_MAX_LOG = 88.7228391


def dry_penalties(history, multiplier, base, allowed_length, breakers, last_n=None):
    window = history if last_n is None else history[-last_n:]
    n = len(window)

    limit = 0
    while limit < n and window[n - 1 - limit] not in breakers:
        limit += 1
    if limit < allowed_length:
        return {}

    rev = window[::-1]
    z = [0] * n
    left = right = 0
    for k in range(1, n):
        if k < right and z[k - left] < right - k:
            z[k] = z[k - left]
            continue
        length = max(0, right - k)
        while k + length < n and rev[length] == rev[k + length]:
            length += 1
        z[k] = length
        left, right = k, k + length

    longest = {}
    for k in range(1, n):
        length = min(z[k], limit)
        if length >= allowed_length:
            candidate = rev[k - 1]
            if candidate not in breakers and length > longest.get(candidate, 0):
                longest[candidate] = length

    max_exponent = int(FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
    penalties = {}
    for candidate, length in longest.items():
        exponent = length - allowed_length
        if 0 < max_exponent < exponent:
            exponent = max_exponent
        penalties[candidate] = multiplier * base**exponent
    return penalties
```

The guard `k + length < n` in the inner loop enforces $m_i \le i + 1$, ensuring an
earlier match never extends past the start of the window. Overlapping matches need no
special case, since the Z-array of `a a a a` is `[0, 3, 2, 1]`.

## Common mistakes

- **Quadratic backward scan.** Walking backward from every earlier occurrence of the
  last token is correct, and it is what the original implementation did (mitigating
  the worst case by capping match length at 50). Without a cap, a 20,000-token
  looping history requires around $10^7$ comparisons per step.
- **Overwriting instead of taking the max per token.** Assigning
  `longest[token] = length` lets whichever occurrence is visited last win. In
  `1 2 3 4 9 3 4 8 1 2 3`, token `4` follows both a match of length 3 and a match of
  length 1, and its penalty must come from the 3.
- **Ignoring breakers or only checking the last token.** Checking `window[-1]` alone
  misses a breaker sitting two tokens from the end, where matches must stop at length
  1 instead of running through the breaker.
- **Penalizing a breaker.** When a breaker follows an earlier match, it must be
  skipped so DRY never blocks the model from ending a line or closing a turn.
- **Counting the candidate in the match length.** The match length counts only tokens
  already in the window. Adding 1 for the candidate shifts every exponent up by one,
  making a match of `allowed_length` cost `multiplier * base` instead of
  `multiplier`.
- **Using `>` instead of `>=`.** A match of exactly `allowed_length` tokens is
  penalized with exponent 0.
- **Forbidding overlapping matches.** Requiring the earlier match to end before the
  trailing match begins cuts the measured length in half for the tightest and most
  common loops (`a a a a`, `a b a b a b`).
- **Applying `last_n` too late.** Slicing `history` to the last `last_n` tokens must
  happen before matching. If you match on the full history and only filter candidate
  positions afterward, matches can extend past the start of the window.
- **Omitting the exponent clamp.** `1.75 ** 3000` raises `OverflowError` in Python
  and overflows to `inf` in float32. Subtracting `inf` from multiple logits before a
  softmax can produce NaNs.
- **Z-box boundary bug.** Copying `z[k - left]` without checking
  `z[k - left] < right - k` is the classic Z-algorithm bug. When the mirrored match
  reaches or passes the end of `[left, right)`, the true match may be longer (or the
  copy may overclaim beyond `right`). This bug often slips past tests without nested
  repeats.

## Complexity

$O(n)$ time and memory in the window length $n$, using one backward pass for `limit`,
the Z-algorithm, and one pass over `z`. The naive scan is $O(n)$ on non-repeating
text but degrades to $O(n^2 / p)$ on a loop of period $p$.

## In the wild

- **The original.** DRY was introduced by p-e-w in text-generation-webui (PR #5677,
  "DRY: A modern repetition penalty that reliably prevents looping"). Its
  `DRYLogitsProcessor` (`modules/sampler_hijack.py`) scans backward from every
  earlier occurrence of the last token, skips when the last token is a breaker,
  skips breaker candidates, stops at breakers inside a match, keeps the maximum match
  per candidate, and applies `multiplier * base ** (match_length - allowed_length)`
  for `match_length >= allowed_length`. The lookback window is controlled by `_range`
  (`input_ids[:, -self._range:]`).
- **llama.cpp.** `llama_sampler_dry_apply` in `src/llama-sampler.cpp` computes the
  same match lengths by running the Z-algorithm backward over the last
  `dry_penalty_last_n` tokens, capped at `rep_limit` (the count of trailing
  non-breakers). This problem follows llama.cpp, including its comment example
  `a b c c b c y a b c` with repeat counts `0 0 3 1 0 2 0 0 0 0`.
- **Differences between the two.** First, for overflow, text-generation-webui caps
  match length at 50 tokens (which also bounds its backward scan), whereas llama.cpp
  leaves match lengths uncapped and clamps the exponent at
  `FLOAT_MAX_LOG / log(base)`. This problem uses llama.cpp's rule. Second,
  text-generation-webui uses single-token breakers, whereas llama.cpp uses string
  breakers (default `\n`, `:`, `"`, `*`) that may span multiple tokens and exempts
  only single-token breakers from penalty. This problem uses single-token breakers,
  where both implementations agree. llama.cpp's defaults (`common/common.h`) are
  multiplier 0 (disabled), base 1.75, and allowed length 2.
- **Position in the chain.** Because DRY modifies logits directly, it runs before
  temperature scaling and truncation samplers (see `sampler-chain`).
