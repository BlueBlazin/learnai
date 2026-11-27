## The idea

A unigram language model scores a sequence of pieces as the product of their probabilities. In log space, the score of a segmentation is the sum of its piece scores, so encoding means finding the segmentation with the largest sum.

A string of $n$ characters has $2^{n-1}$ possible segmentations, but they overlap. Once you pick the first piece, the rest of the segmentation must be the optimal segmentation of the remaining suffix. Otherwise, replacing that suffix with a better one would improve the entire segmentation. Because there are only $n + 1$ suffixes, dynamic programming over suffixes finds the global optimum. This is the Viterbi algorithm on a lattice of matching pieces.

Let $B(i)$ be the best segmentation of `text[i:]`, with $B(n)$ equal to the empty list. For each position $i$ from $n - 1$ down to $0$,

$$
B(i) = \operatorname{best}_{L} \; \big[\, \texttt{text[i:i+L]} \,\big] + B(i + L),
$$

where $L$ ranges over all lengths such that `text[i:i+L]` is in `pieces`, plus $L = 1$ as an unknown character whenever `text[i]` is not in `pieces`.

**Why the tie rules work with suffix DP.** The ranking order is lexicographic on total score, then fewer pieces, then longer pieces from left to right. Prepending the same piece of length $L$ to two segmentations of `text[i+L:]` adds the same score and piece count to both, and places the same length $L$ at the front. That preserves their relative order, so the best segmentation starting with `text[i:i+L]` always continues with $B(i + L)$. Comparing candidates for different lengths $L$ by total score, piece count, and then $L$ itself yields $B(i)$. Filling the table backwards over suffixes makes the third tie rule local. A forward table over prefixes would naturally compare the *last* piece instead of the first.

**Unknown characters.** Without a fallback, any string containing an uncovered character has no valid segmentation. SentencePiece inserts a single-character unknown node whenever no single-character piece matches at position $i$, scoring it at the minimum vocabulary score minus 10. Notice that this condition checks only whether `text[i]` is in `pieces`, not whether *any* piece starts at `i`. For `abcd` with pieces `ab` and `bcd`, a piece starts at `a`, yet choosing unknown `a` followed by `bcd` beats `ab` followed by two unknowns.

## Reference solution (Python)

```python
UNK_PENALTY = 10.0


def unigram_encode(text, pieces):
    n = len(text)
    unk_score = min(pieces.values()) - UNK_PENALTY
    max_len = max(len(piece) for piece in pieces)
    best = [(0.0, 0, 0)] * (n + 1)  # (total, count, length of the first piece)
    for i in range(n - 1, -1, -1):
        choice = None
        for length in range(min(max_len, n - i), 0, -1):
            score = pieces.get(text[i : i + length])
            if score is None:
                if length > 1:
                    continue
                score = unk_score
            rest_score, rest_count, _ = best[i + length]
            total, count = score + rest_score, 1 + rest_count
            if choice is None or total > choice[0] or (total == choice[0] and count < choice[1]):
                choice = (total, count, length)
        best[i] = choice
    out = []
    i = 0
    while i < n:
        length = best[i][2]
        out.append(text[i : i + length])
        i += length
    return out
```

The inner loop tries lengths from longest to shortest and updates `choice` only on a strict improvement, so a tie on total score and piece count keeps the longer first piece. Python's `len(text)` and string slicing operate on Unicode code points, matching the specification.

## Common mistakes

- **Greedy longest match.** Picking the longest piece at each position is WordPiece's rule, not unigram's. On `hugs`, greedy matching picks `hug` ($-9$) and totals $-11$, losing to `hu gs` at $-8.5$. Because long pieces are often high-probability, greedy matching agrees with Viterbi on many common words and hides the bug.
- **Forgetting the penalty.** Scoring an unknown character at the minimum vocabulary score or at a flat $-10$ lets unknowns beat real pieces. With `ab` at $-4$, `c` at $-4$, and `bc` at $-1$, an unpenalized unknown `a` followed by `bc` beats `ab c`.
- **Offering unknowns only as a last resort.** Allowing an unknown character only when no piece starts at `i` fails on `abcd` above. Offering an unknown when the single character *is* in `pieces` is harmless because it always loses to the real piece, but unnecessary.
- **Indexing bytes instead of characters.** Running DP over `text.encode()` matches on covered text, then splits an unknown character like `日` into three penalized unknown bytes that cannot be returned as valid strings.
- **Letting loop order break ties.** Comparing only total scores with `>` returns whichever optimal path the loop visits first, which can use extra pieces or pick the longer *last* piece. With a forward DP over prefixes, preferring the longest first piece requires a second pass, whereas suffix DP handles it in one comparison.
- **Adding a floating-point epsilon.** Real vocabularies have arbitrary floats where exact ties are rare, so this problem uses multiples of $1/8$ to make sums exact in binary floating point. Do not use an epsilon when comparing scores.

## Complexity

Let $n$ be the length of `text` and $L$ be the maximum piece length in `pieces`. The DP table has $n + 1$ entries. Each position checks up to $L$ lengths, and each check slices a string of length up to $L$ and hashes it for a dictionary lookup in $O(L)$ time. This takes $O(n L^2)$ time and $O(n)$ memory. Production encoders walk a trie from each position instead of slicing, finding all matching pieces in $O(L)$ total time per position and reducing the runtime to $O(n L)$.

## In the wild

The unigram model comes from Kudo (*Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*, ACL 2018) and is implemented in SentencePiece (`src/unigram_model.cc`). SentencePiece uses the same constants shown here: `kUnkPenalty = 10.0`, an unknown score of `min_score() - kUnkPenalty`, and an unknown node inserted only when no single-character piece matches (`has_single_node`).

SentencePiece's own Viterbi implementation uses strict `>`, so exact score ties depend on loop order rather than an explicit rule. This problem specifies the tie-breaking order so the output is deterministic. In full SentencePiece models, text is also normalized and spaces are replaced with `▁` before the search, and output pieces are mapped to integer ids afterwards. Replacing the maximum over paths with a sum or with sampling over the same lattice produces the EM training algorithm and subword regularization from the paper.
