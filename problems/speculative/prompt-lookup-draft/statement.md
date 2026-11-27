Implement prompt lookup drafting for speculative decoding. Instead of running a small draft network, search the token context for an earlier occurrence of its trailing tokens and propose the tokens that followed that occurrence.

![This problem is Draft, here copied from an earlier place in the context.](assets/map.svg)

## Specification

```python
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]: ...
```

Let $L$ be the length of `context`, which contains the prompt followed by all tokens generated so far. For an n-gram length $n$, the *suffix* is the last $n$ tokens, `context[L - n :]`. An *earlier occurrence* of that suffix is a start index $s$ that satisfies

$$
0 \le s < L - n
\qquad\text{and}\qquad
\texttt{context}[s : s+n] = \texttt{context}[L-n : L].
$$

The bound $s < L - n$ excludes only the suffix itself. An earlier occurrence may **overlap** the suffix. For example, in `[7, 7, 7]`, the suffix `[7, 7]` has an earlier occurrence at $s = 0$. Because $s < L - n$, at least one token always follows an earlier occurrence.

The function tries n-gram lengths in descending order: `max_ngram`, `max_ngram - 1`, ..., `min_ngram`. It stops at the first $n$ that has an earlier occurrence, selects the **earliest** one (the smallest valid $s$), and returns the tokens that immediately follow it:

$$
\texttt{context}[s+n : s+n+k].
$$

This slice yields `k` tokens, or fewer if `context` ends first (the copy never extends past the end of `context`). Any length $n \ge L$ cannot have an earlier occurrence and is skipped. If no $n$ in the range has an earlier occurrence, return `[]` (in particular, an empty `context` returns `[]`).

- `context`: list of $L \ge 0$ token IDs (Python `int`). Must not be modified.
- `max_ngram`, `min_ngram`: Python `int` values with $1 \le$ `min_ngram` $\le$ `max_ngram`.
- `k`: Python `int` $\ge 1$, the maximum number of draft tokens to return.
- Returns a new list of 0 to `k` Python `int` values.

## Example

```python
lookup_draft([5, 9, 2, 7, 4, 9, 2, 8, 3, 9, 2], 3, 1, 3)   # [7, 4, 9]
lookup_draft([1, 2, 3, 1, 2], 3, 1, 4)                     # [3, 1, 2]
lookup_draft([7, 7, 7], 2, 1, 4)                           # [7]
lookup_draft([4, 6, 5, 8, 6], 3, 2, 4)                     # []
```

- **First call:** For $n = 3$, the suffix `[3, 9, 2]` does not occur earlier. For $n = 2$, the suffix `[9, 2]` occurs at $s = 1$ and $s = 5$. The earliest match is $s = 1$, and the three tokens after it are `context[3:6] = [7, 4, 9]` (matching at $s = 5$ would give `[8, 3, 9]`).
- **Second call:** For $n = 3$, `[3, 1, 2]` has no earlier occurrence. For $n = 2$, `[1, 2]` occurs at $s = 0$. Only three tokens follow it (`context[2:6] = [3, 1, 2]`), so the draft is shorter than `k`.
- **Third call:** For $n = 2$, the suffix `[7, 7]` (indices 1 and 2) also occurs at $s = 0$, overlapping itself. One token follows that occurrence, giving `context[2:6] = [7]`.
- **Fourth call:** Neither `[5, 8, 6]` ($n = 3$) nor `[8, 6]` ($n = 2$) occurs earlier. The 1-gram `[6]` does occur earlier, but `min_ngram` is 2, so the function returns `[]`.

## Constraints

- $0 \le L \le 4096$, $1 \le$ `min_ngram` $\le$ `max_ngram` $\le 8$, and $1 \le k \le 64$. Note that `max_ngram` may exceed $L$.
- The output is compared exactly, including element type (a list of Python `int` values).
- A direct search over every valid start index for each $n$ is fast enough.

## Why this matters

In summarization, retrieval-augmented question answering, and code editing, much of the output copies spans from the prompt. Prompt lookup decoding (Saxena, 2023) exploits this without calling a draft model. It matches the tail of the sequence against earlier tokens and proposes the continuation that followed. Because the target model verifies the draft like any other proposal, the output distribution is unchanged, and copy-heavy workloads run two to four times faster.

This function runs before every target pass, so small indexing bugs hurt acceptance rate silently. Matching the suffix against itself produces an empty draft, preferring shorter n-grams anchors copies on coincidental single tokens, and starting the copy one token off causes every draft token to fail verification.
