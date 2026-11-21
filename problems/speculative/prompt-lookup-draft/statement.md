Implement the draft "model" of prompt lookup decoding: instead of running a small
network, look for the place where the last few tokens of the context occurred
before, and propose the tokens that followed them there. The search is a few lines;
the substance is which occurrence counts (not the suffix itself, the earliest one,
the longest n-gram first) and where the copied tokens start.

![This problem is Draft, here copied from an earlier place in the context.](assets/map.svg)

## Specification

```python
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]: ...
```

Let $L$ be the length of `context` (the prompt followed by everything generated so
far). For an n-gram length $n$, the *suffix* is the last $n$ tokens,
`context[L-n:]`. An *earlier occurrence* of the suffix is a start index $s$ with

$$
0 \le s < L - n
\qquad\text{and}\qquad
\texttt{context}[s : s+n] = \texttt{context}[L-n : L].
$$

The condition $s < L - n$ excludes only the suffix itself. An occurrence may
**overlap** the suffix: in `[7, 7, 7]` the suffix `[7, 7]` has an earlier occurrence
at $s = 0$. An earlier occurrence is always followed by at least one token.

The function tries $n =$ `max_ngram`, `max_ngram - 1`, ..., `min_ngram`, in that
order, and stops at the first $n$ for which an earlier occurrence exists. It takes
the **earliest** one (the smallest $s$) and returns the tokens that followed it:

$$
\texttt{context}[s+n : s+n+k].
$$

That is `k` tokens, or fewer when the context ends first (the copy never runs past
the end of `context`). If no $n$ in the range has an earlier occurrence, the result
is the empty list. Lengths $n \ge L$ cannot have an earlier occurrence and are
skipped; in particular an empty context gives `[]`.

- `context`: a list of $L \ge 0$ token ids (Python ints). It must not be modified.
- `max_ngram`, `min_ngram`: ints with $1 \le$ `min_ngram` $\le$ `max_ngram`.
- `k`: an int $\ge 1$, the maximum number of draft tokens.
- Returns a new list of 0 to `k` Python ints.

## Example

```python
lookup_draft([5, 9, 2, 7, 4, 9, 2, 8, 3, 9, 2], 3, 1, 3)   # [7, 4, 9]
lookup_draft([1, 2, 3, 1, 2], 3, 1, 4)                     # [3, 1, 2]
lookup_draft([7, 7, 7], 2, 1, 4)                           # [7]
lookup_draft([4, 6, 5, 8, 6], 3, 2, 4)                     # []
```

First call. $n = 3$: the suffix `[3, 9, 2]` occurs nowhere earlier. $n = 2$: the
suffix `[9, 2]` occurs at $s = 1$ and at $s = 5$. The earliest occurrence is
$s = 1$, and the three tokens after it are `context[3:6] = [7, 4, 9]`. (The
occurrence at $s = 5$ would have given `[8, 3, 9]`.)

Second call. $n = 3$: `[3, 1, 2]` has no earlier occurrence. $n = 2$: `[1, 2]`
occurs at $s = 0$, and `context[2:6]` holds only three tokens, `[3, 1, 2]`, so the
draft is shorter than `k`.

Third call. $n = 2$: the suffix `[7, 7]` (indices 1 and 2) also occurs at $s = 0$,
overlapping itself. One token follows that occurrence: `context[2:6] = [7]`.

Fourth call. The suffix `[6]` does occur earlier, but `min_ngram` is 2, and neither
`[5, 8, 6]` nor `[8, 6]` occurs earlier: no draft.

## Constraints

- $0 \le L \le 4096$, $1 \le$ `min_ngram` $\le$ `max_ngram` $\le 8$, and
  $1 \le k \le 64$. `max_ngram` may exceed $L$.
- The result is compared exactly, including its type: a list of Python ints.
- A direct search (every start index for every $n$) is fast enough.

## Why this matters

In summarization, retrieval-augmented question answering, and above all code
editing, much of the output is copied from the prompt. Prompt lookup decoding
(Saxena, 2023) exploits that with a draft that costs no model call at all: match the
tail of the sequence against its own past and guess that history repeats. The target
model verifies the guess like any other draft, so the output is unchanged, and on
copy-heavy inputs decoding gets two to four times faster. This function runs before
every target pass, so its details show up directly in the acceptance rate: matching
the suffix with itself yields no draft at all, preferring a short n-gram anchors the
copy on a coincidence, and starting the copy one token off makes every guess wrong.
