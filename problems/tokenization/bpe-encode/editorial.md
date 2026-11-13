## The idea

BPE training produces merges one at a time: "the most frequent adjacent pair is
`(e, r)`, call it a new token; now the most frequent pair is ...". Encoding replays
that history on new text. The merge learned first must be applied first, because
later merges were learned on text in which the earlier ones had already happened.
That is why the rule is "the lowest-rank pair present in the sequence", and why two
tempting shortcuts are wrong:

- *Scan the text and merge whatever pair you meet first.* For `abc` with `(b, c)` at
  rank 0 and `(a, b)` at rank 1 this produces `[ab] c`, but training merged `b c`
  everywhere before it ever counted `a b`, so the model only knows `a [bc]`.
- *Find the longest vocabulary entry at each position.* That is WordPiece. With merges
  `(b, c)`, `(a, b)`, `([ab], c)` the vocabulary contains `abc`, yet BPE can never
  produce it from the text `abc`: after rank 0 there is no `[ab]` to extend.

The loop terminates because every applied merge shortens the sequence by at least one.
The result is unique because the pairs are distinct: exactly one pair has the lowest
rank, and step 4 of the statement fixes what happens to its occurrences.

Within one pass, occurrences can only collide when the pair is `(x, x)` and the text
has a run of three or more `x`. Consuming both positions of a match (`i += 2`) resolves
it left to right.

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

`decode` uses a stack instead of recursion so that a deep chain of merges (token 900
built from token 899 built from ...) cannot hit Python's recursion limit, and it looks
merges up by id on demand, so it does not care in which order the table lists them.

## Why bytes

Starting from bytes gives a closed base vocabulary of 256 symbols that covers every
possible input: no unknown-token id, no normalization table, any emoji or script works
on day one. The price is that tokens are byte strings, not strings. A token can be the
first two bytes of a three-byte character. So `decode` must join bytes first and
convert to text once, and a streaming server must hold back a trailing incomplete
character until the next token completes it.

## Common mistakes

- **One sweep over the merge list.** `for i, pair in enumerate(merges): replace(pair)`
  is correct only if no merge needs a token created by a later one. A table straight
  from training has that property; the statement does not promise it, and the case
  with the merges `[(257, c), (a, b)]` shows the difference: rank 0 becomes applicable
  only after rank 1 ran, and a single sweep has already moved past it.
- **Overlapping matches.** Marking every position where the pair starts and merging at
  all of them turns `a a a` into two `[aa]` tokens, which decode to four bytes. The
  round trip fails only on runs, so ordinary English text hides the bug.
- **Characters instead of bytes.** `[ord(ch) for ch in text]` agrees with UTF-8 on
  ASCII and nowhere else: `é` becomes the single id 233 instead of `195, 169`, and code
  points above 255 collide with merge ids.
- **Merging the leftmost pair.** Picking the first mergeable pair in the text instead
  of the lowest-ranked one anywhere (see above).
- **Greedy longest match.** A different tokenization algorithm (see above).
- **Decoding token by token.** `"".join(bytes_of(t).decode(...) for t in ids)` breaks
  every character whose bytes straddle a token boundary, and with `errors="replace"`
  it does so silently.

## Complexity

Let $n$ be the number of bytes and $m$ the number of merges. Each pass costs $O(n)$
and removes at least one token, so there are at most $n - 1$ passes and the reference
takes $O(n^2)$ time in the worst case and $O(n + m)$ memory. With a table that only
refers to earlier merges, every merge is applied in at most one pass, which tightens
the bound to $O(n \cdot \min(n, m))$. On real text the number of passes is far below
both bounds, but it does grow with the input length.

Faster approaches, when it matters:

- **Sweep the merge list once** when the table is known to refer only to earlier
  merges. Then applying a merge can never make a lower-ranked one applicable, and the
  single sweep is exact: $O(n \cdot m)$ in the worst case but trivially simple, and
  fast if you skip merges whose pair does not occur.
- **Priority queue over a linked list.** Keep the tokens in a doubly linked list and
  the mergeable positions in a heap keyed by (rank, position). A merge touches only its
  two neighbors, so the whole encoding is $O(n \log n)$. Hugging Face `tokenizers` does
  this per word.
- **Pre-tokenize and cache.** GPT-2 splits text into words with a regular expression
  first and encodes each word on its own; words repeat, so a dictionary from word to
  ids removes most of the work. Splitting also bounds $n$ per call.

## In the wild

GPT-2's original `encoder.py` is this algorithm on strings of byte-characters: its
`bpe()` function computes `min(pairs, key=lambda pair: self.bpe_ranks.get(pair,
float("inf")))` in a loop. `tiktoken` keys the table by the merged bytes and uses the
token id as the rank, with the same "lowest rank first" loop in Rust. `llama.cpp`
implements BPE for GPT-2-style vocabularies with a priority queue over symbol pairs.
All of them add a pre-tokenization step and special tokens on top, which are left out
here.
