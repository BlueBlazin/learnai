## The idea

DRY asks one question per decoding step: *if I emit token $x$ now, how long is the
verbatim repeat it would extend?* The history ends in some run of tokens. Wherever
that run occurred before, the token that came next is the one that would continue
the repeat, and the length of the shared run measures how deep into a loop the
generation already is. The penalty $\text{multiplier} \cdot \text{base}^{M -
\text{allowed}}$ is zero for short repeats (names, idioms, list markers shorter
than `allowed_length`), mild at the threshold, and multiplies by `base` with every
further token, so no loop survives more than a few tokens, whatever the logits.

Two refinements make it usable:

- **Sequence breakers.** Chat templates, newlines, and quotation marks repeat
  legitimately and separate units of text. A match may not contain a breaker, and a
  breaker is never penalized. Since a match is always a run at the *end* of the
  window, "may not contain a breaker" is the same as "is at most as long as the
  run of non-breakers at the end of the window": one number, `limit`, computed
  once.
- **The longest match per token.** A token can follow several earlier occurrences
  of the current suffix; the longest one decides.

**The matching.** The match length at position $i$ is the length of the longest
common suffix of $w_0 \ldots w_i$ and the whole window. Reverse the window and it
becomes the longest common prefix of `rev` and `rev[k:]` with $k = n - 1 - i$: the
**Z-array**. The Z-algorithm computes all of it in $O(n)$ by remembering the
rightmost matched stretch `[left, right)`: inside it, `rev[k:]` looks like
`rev[k - left:]`, whose answer is already known. If that answer ends before the
stretch does, it is the answer for $k$ too. Otherwise the match reaches at least to
`right`, and only there do new comparisons start. Each successful comparison
advances `right`, which never moves back, so the total work is linear.

The naive scan compares backward from every position. On ordinary text that is
fine, since almost every comparison fails at once. On a loop of period $p$ every
$p$-th position matches all the way back to the start of the loop, and the cost is
$n^2 / (2p)$.

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

`k + length < n` in the inner loop is what bounds $m_i \le i + 1$: the earlier run
may not start before the window. Overlap needs no code: the Z-array of `a a a a`
is `[_, 3, 2, 1]`.

## Common mistakes

- **The quadratic scan.** Correct, and the first thing everyone writes: for each
  earlier occurrence of the last token, walk backward while the tokens agree. It
  is the original implementation (see below), which bounds the damage by capping
  the match length at 50. Without a cap, a 20,000-token history in a loop costs
  $10^7$ comparisons per step.
- **Not keeping the longest match per token.** Writing `longest[token] = length`
  lets whichever occurrence is visited last decide. In `1 2 3 4 9 3 4 8 1 2 3`
  the token `4` follows a match of length 3 and one of length 1; the penalty must
  come from the 3.
- **Ignoring breakers**, or checking only the last token. The second is subtle:
  with the breaker two tokens from the end, matches must stop at length 1, but a
  check of `window[-1]` alone lets them run through it.
- **Penalizing a breaker.** A breaker can be the candidate (the token that
  followed an earlier occurrence). It is exempt: DRY must never stop the model
  from ending a line or closing a turn.
- **Counting the candidate.** The match length is the number of tokens already in
  the window. Adding one for the token about to be emitted shifts every exponent,
  and a match of exactly `allowed_length` then costs `multiplier * base` instead of
  `multiplier`.
- **`>` instead of `>=`.** A match of exactly `allowed_length` tokens is penalized
  (with exponent 0).
- **Forbidding overlap.** Requiring the earlier occurrence to end before the
  current run starts halves the measured length of the tightest loops (`a a a a`,
  `a b a b a b`), which are the commonest.
- **The window applied late.** `last_n` cuts the history *before* matching. If
  matches are measured on the full history and only the candidates are filtered
  by position, a match extends past the start of the window.
- **No exponent clamp.** `1.75 ** 3000` raises `OverflowError` in Python and is
  `inf` in float32, and `logit - inf` for the one token the loop wants is fine,
  but `inf` penalties on several tokens followed by a softmax are how NaNs start.
- **A Z-box bug.** Copying `z[k - left]` without checking it against `right - k`
  is the classic error in the algorithm: when the mirrored match reaches the end
  of the stretch, the true match may be longer (or the copy may claim more than
  was ever compared). It passes every test that has no nested repeats.

## Complexity

$O(n)$ time and memory in the window length: one pass for `limit`, the
Z-algorithm, and one pass over the Z-array. The naive scan is $O(n)$ on text
without repeats and $O(n^2 / p)$ on a loop of period $p$.

## In the wild

- **The original.** DRY was introduced by p-e-w in text-generation-webui (pull
  request #5677, "DRY: A modern repetition penalty that reliably prevents
  looping"). Its `DRYLogitsProcessor` (`modules/sampler_hijack.py`) finds every
  earlier occurrence of the last token and extends each match backward token by
  token: skip when the last token is a breaker, skip a candidate that is a
  breaker, stop at a breaker inside the match, keep the maximum per candidate,
  and penalize by `multiplier * base ** (match_length - allowed_length)` when
  `match_length >= allowed_length`. The lookback is its `_range` argument
  (`input_ids[:, -self._range:]`).
- **llama.cpp** (`llama_sampler_dry_apply` in `src/llama-sampler.cpp`) computes
  the same match lengths with the Z-algorithm run backward over the last
  `dry_penalty_last_n` tokens, capped at `rep_limit`, the number of tokens after
  the last breaker; this problem follows it, including the comment's example
  `a b c c b c y a b c` with repeat counts `0 0 3 1 0 2 0 0 0 0`.
- **Where the two differ.** (1) *Overflow*: the webui caps the match length at 50
  tokens ("to prevent exponent overflow", which also bounds its scan);
  llama.cpp leaves lengths uncapped and clamps the exponent at
  `FLOAT_MAX_LOG / log(base)`. This problem uses llama.cpp's rule. (2) *Breakers*:
  the webui's breakers are single tokens; llama.cpp's are strings (default `\n`,
  `:`, `"`, `*`) that may tokenize to several tokens, matched as head token plus
  tail, and only single-token breakers are exempt from the penalty. This problem
  has single-token breakers, where the two agree. llama.cpp's defaults
  (`common/common.h`) are multiplier 0 (disabled), base 1.75, and allowed length
  2.
- **Position in the chain.** DRY edits logits, so it runs before temperature and
  the truncation samplers (`sampler-chain`).
