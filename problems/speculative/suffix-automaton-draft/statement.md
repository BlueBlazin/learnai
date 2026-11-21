Implement the index behind prompt-lookup drafting at scale: a structure that, as the
sequence grows one token at a time, always knows the **longest** suffix that has
occurred before and where it occurred **first**, so that the tokens which followed
it can be proposed as a draft. Searching the history again for every draft is easy
and quadratic; a suffix automaton built online answers in constant time per token.

![This problem is Draft, here read from an index of the sequence so far.](assets/map.svg)

## Specification

```python
class SuffixIndex:
    def __init__(self) -> None: ...
    def extend(self, token: int) -> None: ...
    def match_length(self) -> int: ...
    def draft(self, max_len: int, min_match: int = 1) -> list[int]: ...
```

The index holds a sequence $x_0, \dots, x_{L-1}$: every token passed to `extend`,
in order. A decoder feeds it the prompt and then each token it emits.

**The match.** For $1 \le n < L$, the suffix of length $n$ *occurred earlier* if
there is a start index $s$ with

$$
0 \le s < L - n
\qquad\text{and}\qquad
x_{s..s+n-1} = x_{L-n..L-1} .
$$

As in `prompt-lookup-draft`, only the suffix itself is excluded: an earlier
occurrence may overlap it. The *match length* $m$ is the largest such $n$, or 0
when no suffix occurred earlier (in particular for $L \le 1$). There is no upper
bound on $m$ other than $L - 1$. The *match position* is the **smallest** $s$ for
$n = m$: the earliest occurrence of the longest match.

- `extend(token)` appends `token` (an int $\ge 0$) and returns `None`.
- `match_length()` returns $m$ for the current sequence, a Python `int`.
- `draft(max_len, min_match=1)` returns a new list:
  - `[]` if $m <$ `min_match`;
  - otherwise the tokens that followed the earliest occurrence of the match,
    $x_{s+m}, \dots$, at most `max_len` of them and never past the end of the
    sequence: `x[s + m : s + m + max_len]`.

  `max_len` and `min_match` are ints $\ge 1$. Since $s + m < L$, a non-empty match
  always yields at least one token. Neither query changes the index, and the
  returned list is the caller's to modify.

Calls may be interleaved in any order: any number of queries between two `extend`
calls, and `extend` again after a query. Separate `SuffixIndex` objects are
independent.

**Cost.** `extend` must run in amortized constant time (for a bounded number of
distinct tokens following any one context; a hash map per state is fine),
`match_length` in constant time, and `draft` in time proportional to the length of
its result. The speed case feeds 12,000 tokens and asks for a draft after each one;
the solution must be within 5x of the reference.

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

At the end, the suffix `[3, 9, 2]` has not occurred earlier, and `[9, 2]` has, at
$s = 1$ and at $s = 5$: $m = 2$. The earliest occurrence is $s = 1$, and the
tokens after it are `x[3:6] = [7, 4, 9]`. (The occurrence at $s = 5$ would give
`[8, 3, 9]`.) With `min_match = 3` the match is too short to draft from.

```python
index = SuffixIndex()
for token in [7, 7, 7]:
    index.extend(token)

index.match_length()  # 2: [7, 7] at indices 1..2 also occurs at s = 0, overlapping itself
index.draft(4)  # [7]: one token follows that occurrence
```

## Constraints

- Up to $10^5$ calls of `extend` per index; token ids in $[0, 2^{20})$; `max_len`
  up to 64.
- Results are compared exactly, including the types: a Python `int`, and a list of
  Python `int`s.
- The speed limit is relative to the reference on the same machine: 5x on a
  12,000-token sequence with one `draft(8, 2)` per token. Searching the history
  again for each draft, even with a fast substring search, measures about 50x.
- Pure Python; the standard library is available, and nothing in it does this for
  you.

## Why this matters

Prompt lookup decoding costs no model call, so it is the first speculation method
an engine turns on for copy-heavy work: code edits, retrieval-augmented answers,
summaries. Its plain form matches an n-gram of two or three tokens by scanning the
context, which has two limits. A short n-gram anchors the copy on a coincidence,
and the scan grows with the context, in the inner loop of every decoding step, on
the CPU thread that also has to keep the GPU fed. SAM-Decoding replaces the scan
with a suffix automaton that is extended as tokens are generated: the match is the
longest one that exists, its length tells the engine whether the draft is worth
verifying at all, and the cost per token does not depend on how long the context
has become. The structure is unforgiving, though. A split state that forgets its
earliest position, or is never split, still returns plausible drafts, only from the
wrong place: acceptance drops, nothing crashes, and the cause is invisible unless
the index is tested against the definition.
