## The idea

BPE training learns merges one at a time by repeatedly merging the most frequent adjacent pair into a new token. Encoding replays that merge history on new text. Earlier merges must be applied first because later merges were learned on text where earlier merges had already taken place.

This explains why the rule always selects the lowest-rank pair present in the sequence, and why two common shortcuts fail:

- **Merging the leftmost pair.** Scanning the sequence and merging the first valid pair fails on `"abc"` with `(b, c)` at rank 0 and `(a, b)` at rank 1. It produces `[ab] c`, whereas training merged `b c` everywhere before counting `a b`, leaving `a [bc]`.
- **Greedy longest match.** Picking the longest vocabulary entry at each position is WordPiece, not BPE. With merges `(b, c)`, `(a, b)`, and `([ab], c)`, the vocabulary contains `abc`, yet BPE never produces it from `"abc"` because rank 0 consumes `b c` before `[ab]` can form.

The encoding loop always terminates because every applied merge shortens the sequence by at least one token. The output is unique because all pairs in `merges` are distinct, so one pair has the lowest rank at each step, and the left-to-right pass fixes how its occurrences are replaced.

Within a single pass, matches can only overlap when the chosen pair is `(x, x)` and the sequence contains a run of three or more `x`. Advancing the index by two (`i += 2`) after each match resolves overlaps from left to right.

## Reference solution (Python)

```python
from itertools import pairwise


def encode(text, merges):
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(pairwise(ids), key=lambda p: rank.get(p, len(rank)))
        if pair not in rank:
            break
        ids = _replace_pair(ids, pair, 256 + rank[pair])
    return ids


def _replace_pair(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def decode(ids, merges):
    data = bytearray()
    stack = list(reversed(ids))
    while stack:
        token = stack.pop()
        if token < 256:
            data.append(token)
        else:
            left, right = merges[token - 256]
            stack.append(right)
            stack.append(left)
    return data.decode("utf-8", errors="replace")
```

`decode` uses an explicit stack instead of recursion so a deep merge chain cannot hit Python's recursion limit. It also looks up merges by id on demand, so the order of entries in `merges` does not matter.

## Why bytes

Starting from UTF-8 bytes gives a closed base vocabulary of 256 symbols that covers every possible input. No unknown-token id or normalization table is needed, and any emoji or script works immediately.

The tradeoff is that tokens represent arbitrary byte strings rather than complete Unicode characters. A single token can hold only the first two bytes of a three-byte character. That is why `decode` must concatenate all bytes before decoding once, and why a streaming server must buffer an incomplete trailing character until the next token arrives.

## Common mistakes

- **One sweep over the merge list.** Running `for i, pair in enumerate(merges): replace(pair)` works only when every merge refers solely to earlier merges. A table straight from training satisfies that property, but this problem does not guarantee it. With `[(257, c), (a, b)]`, rank 0 becomes applicable only after rank 1 runs, so a single sweep misses it.
- **Overlapping matches.** Marking every starting index where the pair occurs and replacing all of them turns `a a a` into two `[aa]` tokens, which decode to four bytes. Because this bug only appears on repeated runs, ordinary English text often hides it.
- **Characters instead of bytes.** Using `[ord(ch) for ch in text]` matches UTF-8 on ASCII and breaks everywhere else. `"é"` becomes id `233` instead of `195, 169`, and code points above 255 collide with merge ids.
- **Merging the leftmost pair.** Picking the first mergeable pair in the text instead of the lowest-ranked pair anywhere in the sequence produces the wrong segmentation.
- **Greedy longest match.** Matching the longest vocabulary entry at each position implements WordPiece rather than ranked BPE.
- **Decoding token by token.** Running `"".join(bytes_of(t).decode(...) for t in ids)` breaks every character split across a token boundary. With `errors="replace"`, this failure happens silently.

## Complexity

Let $n$ be the number of UTF-8 bytes in `text` and $m$ be the number of merges. Each pass takes $O(n)$ time and removes at least one token, so there are at most $n - 1$ passes. The reference solution therefore runs in $O(n^2)$ worst-case time and $O(n + m)$ space. When merges only reference earlier merges, each merge applies in at most one pass, tightening the time bound to $O(n \cdot \min(n, m))$.

Faster approaches when performance matters:

- **Single sweep over the merge list.** When the table is guaranteed to refer only to earlier merges, applying a merge never enables a lower-ranked one. A single pass over `merges` is then exact, taking $O(n \cdot m)$ worst-case time while staying simple.
- **Priority queue over a doubly linked list.** Store tokens in a doubly linked list and mergeable positions in a min-heap keyed by `(rank, position)`. Each merge updates only its two neighbors, giving $O(n \log n)$ time. Hugging Face `tokenizers` uses this approach per word.
- **Pre-tokenization and caching.** GPT-2 splits text into words with a regular expression and encodes each word separately. Caching word-to-id mappings avoids repeating work on common words and keeps $n$ small per call.

## In the wild

GPT-2's original `encoder.py` uses this exact loop over byte-character strings, computing `min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))` at each step. `tiktoken` keys its table by merged byte strings and uses token ids as ranks, running the same lowest-rank-first loop in Rust. `llama.cpp` implements GPT-2-style BPE with a priority queue over symbol pairs. Production tokenizers also add pre-tokenization and special token handling on top of this core loop.
