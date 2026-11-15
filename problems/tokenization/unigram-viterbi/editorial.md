## The idea

A unigram language model assigns a sequence of pieces the product of their
probabilities, so the log-probability of a segmentation is the sum of its piece
scores, and tokenizing means finding the segmentation with the largest sum. A text of
$n$ characters has $2^{n-1}$ segmentations, but they share structure: once the first
piece is chosen, what follows is a segmentation of the remaining suffix, and it must
be the best one, since replacing it with a better one would improve the whole. There
are only $n + 1$ suffixes, so a table over them suffices. This is the Viterbi
algorithm on a lattice whose nodes are the pieces that match at each position.

Let $B(i)$ be the best segmentation of `text[i:]`, with $B(n)$ empty. Then

$$
B(i) = \operatorname{best}_{L} \; \big[\, \texttt{text[i:i+L]} \,\big] + B(i + L),
$$

over the lengths $L$ for which `text[i:i+L]` is a vocabulary piece, plus $L = 1$ as an
unknown character when `text[i]` is not in the vocabulary.

**Why the tie rules survive the recursion.** The order is lexicographic: total score,
then fewer pieces, then longer pieces from the left. Prepending the same piece to two
suffix segmentations adds the same score and the same count of one to both and puts
the same length in front, so it preserves their order. Hence the best segmentation
that starts with a given piece continues with $B(i + L)$, and comparing the candidates
for different $L$ on (total, count) and then on $L$ itself gives $B(i)$. This is also
why the table runs over suffixes. Over prefixes, the natural tie rule is about the
*last* piece, and "longest first piece" cannot be decided locally.

**Unknown characters.** Without a fallback, a text containing a character that no
piece covers has no segmentation at all. SentencePiece adds one node per such
character, scored below every real piece: the minimum score in the vocabulary minus a
penalty of 10. The condition for adding it is local to the character: no
single-character piece matches there. It is not "nothing matches here". With pieces
`ab` and `bcd` and the text `abcd`, a piece does start at `a`, and the best
segmentation is still the unknown `a` followed by `bcd`, because the alternative pays
for two unknowns.

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

Lengths are tried from the longest down and a later candidate must be strictly
better, so a tie on total and count keeps the longer first piece. `len(text)` and
slicing count code points, which is the unit the statement asks for.

## Common mistakes

- **Greedy longest match.** Taking the longest piece at each position is WordPiece's
  rule, not this model's. On `hugs` it picks `hug` at $-9$ and ends at $-11$, against
  $-8.5$ for `hu gs`. It agrees with Viterbi whenever long pieces are also likely
  ones, which is most of the time, and that is what makes it a dangerous shortcut.
- **Forgetting the penalty.** Scoring an unknown at the minimum score, or at $-10$
  flat, makes it competitive with real pieces: with `ab` at $-4$, `c` at $-4$, and
  `bc` at $-1$, an unpenalized unknown `a` followed by `bc` beats `ab c`.
- **The unknown as a last resort.** Offering it only where no piece starts misses
  `abcd` above, and offering it for characters that are in the vocabulary is merely
  useless (it always loses to the character's own piece).
- **Bytes instead of characters.** Running the table over `text.encode()` gives the
  same pieces for text the vocabulary covers, and splits an unknown `日` into three
  unknown bytes, each penalized and none of them a string that concatenates back.
- **Ties decided by loop order.** Comparing totals only, with `>`, returns whichever
  optimal segmentation the loop meets first: more pieces than necessary in one loop
  order, the longer *last* piece in another. With a forward table over prefixes, the
  rule "longest first piece" needs a second pass; over suffixes it is one comparison.
- **Comparing floats that should be equal.** Real vocabularies have arbitrary
  scores, where two paths rarely tie exactly; the tests use multiples of $1/8$ so
  that a tie is a tie in any order of addition. Do not add an epsilon.

## Complexity

With $n$ characters and pieces of at most $L$ characters, the table has $n$ entries
and each tries at most $L$ lengths, each a slice and a dictionary lookup of $O(L)$:
$O(n L^2)$ time in this form and $O(n)$ memory. Production encoders walk a trie from
each position instead of slicing, which finds all matching pieces in $O(L)$ and makes
the whole search $O(nL)$.

## In the wild

The model is Kudo's (*Subword Regularization: Improving Neural Network Translation
Models with Multiple Subword Candidates*, ACL 2018), and the implementation is
SentencePiece, `src/unigram_model.cc`. Its constants are the ones used here:
`kUnkPenalty = 10.0`, the unknown score `min_score() - kUnkPenalty`, and an unknown
node inserted at a position only when no single-character piece matched there
(`has_single_node`). Its Viterbi compares candidates with a strict `>`, so equal
totals are resolved by the order in which its loops visit nodes, not by a stated
rule; this problem states one so that the answer is defined. Real SentencePiece
models also normalize the text and replace spaces with `▁` before the search, and map
the pieces to ids afterwards; both steps are outside this function. The same lattice,
with the maximum replaced by a sum or by sampling, gives the training algorithm and
subword regularization of the paper.
