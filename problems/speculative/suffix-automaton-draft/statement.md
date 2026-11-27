Implement the online index behind scalable prompt-lookup drafting. As a token sequence grows one token at a time, the index tracks the **longest** suffix that occurred earlier in the sequence and where it occurred **first**, so the tokens that followed that earlier occurrence can be proposed as a draft. Rescanning the history for every draft takes quadratic time, whereas an online suffix automaton answers each query in constant time per token.

![This problem is Draft, here read from an index of the sequence so far.](assets/map.svg)

## Specification

```python
class SuffixIndex:
    def __init__(self) -> None: ...
    def extend(self, token: int) -> None: ...
    def match_length(self) -> int: ...
    def draft(self, max_len: int, min_match: int = 1) -> list[int]: ...
```

The index stores the sequence $x_0, \dots, x_{L-1}$ of all $L$ tokens passed to `extend`, in order. A decoder feeds it the prompt tokens and then each generated token.

**Longest earlier suffix match.** For $1 \le n < L$, the suffix of length $n$ *occurred earlier* if there is a start index $s$ such that

$$
0 \le s < L - n
\qquad\text{and}\qquad
x_{s..s+n-1} = x_{L-n..L-1} .
$$

As in `prompt-lookup-draft`, only the suffix itself (starting at $L - n$) is excluded, so an earlier occurrence may overlap the suffix. The *match length* $m$ is the largest such $n$, or $0$ when no suffix occurred earlier (including when $L \le 1$). There is no upper bound on $m$ other than $L - 1$. The *match position* $s$ is the **smallest** valid start index for $n = m$, giving the earliest occurrence of that longest match.

- `extend(token)` appends `token` (a Python `int` $\ge 0$) to the sequence and returns `None`.
- `match_length()` returns $m$ for the current sequence as a Python `int`.
- `draft(max_len, min_match=1)` returns a new list of Python `int`s:
  - `[]` if $m < \text{min\_match}$;
  - otherwise `x[s + m : s + m + max_len]`, the tokens that followed the earliest occurrence of the match, capped at `max_len` tokens and never extending past the end of the sequence.

Both `max_len` and `min_match` are ints $\ge 1$. Because $s + m < L$, any non-empty match has at least one token following it. Neither `match_length` nor `draft` modifies the index, and callers may mutate the returned list without affecting the index.

Calls may be interleaved in any order, with any number of queries between `extend` calls and further `extend` calls after queries. Separate `SuffixIndex` instances must be completely independent.

**Cost.** `extend` must run in amortized $O(1)$ time (assuming a bounded number of distinct tokens follow any given context; a per-state dictionary is fine). `match_length` must run in $O(1)$ time, and `draft` must run in $O(\text{len(result)})$ time. The speed benchmark appends 12,000 tokens and requests a draft after each one. Your implementation must finish within 5x of the reference time.

## Example

```python
index = SuffixIndex()
for token in [5, 9, 2, 7, 4, 9, 2, 8, 3, 9, 2]:
    index.extend(token)

index.match_length()  # 2
index.draft(3)  # [7, 4, 9]
index.draft(3, 2)  # [7, 4, 9]
index.draft(3, 3)  # []
```

The match length after each `extend`, token by token:

| token | 5 | 9 | 2 | 7 | 4 | 9 | 2 | 8 | 3 | 9 | 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $m$ | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 1 | 2 |

![The example's first seven tokens: each state shows its longest length, and the last state's dashed suffix link ends at $m = 2$. Links to the start are not drawn.](assets/automaton.svg)

After the final token, `[3, 9, 2]` has not appeared earlier, while `[9, 2]` appeared at $s = 1$ and $s = 5$. Thus $m = 2$, the earliest occurrence is $s = 1$, and the tokens following it are `x[3:6] = [7, 4, 9]`. (The later occurrence at $s = 5$ would give `[8, 3, 9]`.) With `min_match = 3`, the match is too short and `draft` returns `[]`.

An overlapping match works the same way:

```python
index = SuffixIndex()
for token in [7, 7, 7]:
    index.extend(token)

index.match_length()  # 2: [7, 7] at indices 1..2 also occurs at s = 0, overlapping itself
index.draft(4)  # [7]: one token follows that occurrence
```

## Constraints

- Up to $10^5$ calls to `extend` per index.
- Token ids are in $[0, 2^{20})$, and `max_len` is at most 64.
- Return values are compared exactly, including types (a Python `int` for `match_length`, and a `list` of Python `int`s for `draft`).
- The speed limit is 5x the reference runtime on the same machine over a 12,000-token sequence with one `draft(8, 2)` call per token. Rescanning the history on every draft, even with a fast substring search, runs about 50x slower.
- Pure Python only. The standard library is available, though nothing in it builds a suffix automaton for you.

## Why this matters

Prompt lookup decoding requires no draft model call, making it the first speculation method an inference engine enables for copy-heavy workloads such as code edits, retrieval-augmented generation, and summarization. Its basic version scans the context for a short 2- or 3-token n-gram, which has two drawbacks. Short n-grams often match by coincidence, and the scan grows with context length inside the CPU decoding loop that must keep the GPU fed.

SAM-Decoding replaces the scan with an online suffix automaton updated as tokens arrive. It always finds the longest existing match, uses the match length to decide whether drafting is worthwhile, and runs in constant time regardless of context length. However, subtle bugs in state splitting fail silently. A cloned state that forgets its earliest end position, or a missing split, still returns plausible token sequences from the wrong location. Acceptance rates drop without any crash, and the bug stays hidden unless the index is tested directly against its specification.
