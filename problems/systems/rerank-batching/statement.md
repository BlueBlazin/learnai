A reranker (a cross-encoder) reads a query and a document together and returns one
relevance score. Score a list of candidate documents against a query in as few padded
tokens as possible: sort the pairs by length, cut the sorted list into batches under
two limits, and put every score back where its document was. The work is in the
bookkeeping, because the batches are scored in an order the caller never sees.

## Specification

```python
def rerank(
    query: list[int],
    docs: list[list[int]],
    scorer,
    max_batch: int,
    top_n: int,
    max_tokens: int | None = None,
) -> tuple[list[float], list[int]]: ...
```

**The scorer.** `scorer` is a kit `CountingScorer`. A *pair* is a tuple
`(query_tokens, doc_tokens)` of two non-empty sequences of ints, and
`scorer(pairs)` takes a non-empty list of pairs and returns a list with one float
per pair, in the order given. The score of a pair depends on that pair alone: it is
the same float whatever else is in the batch. The scorer raises `ValueError` for an
empty list and for a pair whose length `len(query_tokens) + len(doc_tokens)` exceeds
`scorer.max_length`.

A batch is processed as a rectangle: every pair is padded to the longest pair of
its batch. The scorer charges for it,

$$
\text{padded tokens of a call} = (\text{number of pairs}) \times (\text{length of the longest pair}),
$$

and adds that to `scorer.counters.padded_tokens`; `scorer.counters.calls` counts the
calls.

**What `rerank` does**, with $q$ = `len(query)`:

1. **Truncate.** Document $i$ is cut to its first `scorer.max_length - q` tokens if
   it is longer. The query is never shortened. Pair $i$ is `(query, truncated doc i)`
   and its length is $L_i = q + \text{len(truncated doc } i)$.
2. **Sort.** Order the pairs by $L_i$, shortest first. Pairs of equal length stay in
   their original order.
3. **Batch.** Walk the sorted pairs and fill one batch at a time. The next pair $p$
   joins the current batch of $m$ pairs unless that would break a limit:
   - $m + 1 >$ `max_batch` (too many pairs), or
   - `max_tokens` is not `None` and $(m + 1) \cdot L_p >$ `max_tokens` (the rectangle
     would be too large; $L_p$ is its width, since $p$ is the longest pair so far).

   If it would, the current batch is closed and $p$ starts a new one.
4. **Score.** Call `scorer` once per batch. Every pair is scored exactly once.
5. **Restore.** `scores[i]` is the score of `docs[i]`, as the scorer returned it.
6. **Rank.** `ranking` lists the indices of the `min(top_n, len(docs))` documents
   with the highest scores, best first. Documents with equal scores are ranked by
   index, lower first.

The function returns the tuple `(scores, ranking)`: a list of `len(docs)` floats and
a list of ints. For an empty `docs` it returns `([], [])` without calling the scorer.
It does not modify `query`, `docs`, or any document.

## Example

```python
query = [7, 3]  # q = 2
docs = [[5], [9, 9, 9, 9, 9, 9], [4, 8], [6, 6, 6, 6, 6, 6], [2]]
#  L =   3          8              4            8             3
scores, ranking = rerank(query, docs, scorer, max_batch=4, top_n=2, max_tokens=12)
```

Sorted by length, the pairs are documents `0, 4, 2, 1, 3` with lengths
`3, 3, 4, 8, 8`.

| step | pair | $m$ | $(m + 1) \cdot L_p$ | decision |
|---|---|---|---|---|
| 1 | doc 0, $L = 3$ | 0 | 3 | starts batch 1 |
| 2 | doc 4, $L = 3$ | 1 | 6 | joins |
| 3 | doc 2, $L = 4$ | 2 | 12 | joins (12 is not over 12) |
| 4 | doc 1, $L = 8$ | 3 | 32 | over `max_tokens`: starts batch 2 |
| 5 | doc 3, $L = 8$ | 1 | 16 | over `max_tokens`: starts batch 3 |

Three calls, with $3 \cdot 4 + 1 \cdot 8 + 1 \cdot 8 = 28$ padded tokens. Batching the
documents as they arrive under the same limits gives `[0]`, `[1]`, `[2]`, `[3]`,
`[4]`: five calls. Without the token limit (`max_tokens=None`) and with
`max_batch=3`, the sorted batches are `[0, 4, 2]` and `[1, 3]`, 28 padded tokens in
two calls; arrival order gives `[0, 1, 2]` and `[3, 4]`, which is
$3 \cdot 8 + 2 \cdot 8 = 40$.

Whatever the batches, `scores[1]` is the score of `docs[1]`, the first document of
the second call, and not the second score that came back.

## Constraints

- $1 \le q <$ `scorer.max_length`, so at least one document token always remains.
  Documents are not empty. `max_batch` $\ge 1$, `top_n` $\ge 0$, and when
  `max_tokens` is given it is at least `scorer.max_length`, so every pair fits in a
  batch by itself. At most 500 documents.
- Scores are compared exactly with the scorer's own score of each truncated pair,
  and `ranking` exactly.
- **Cost.** `scorer.counters.padded_tokens` and `scorer.counters.calls` must not
  exceed what the rule above produces, `scorer.counters.pairs` must equal
  `len(docs)`, and no call may break `max_batch` or `max_tokens`. With
  `max_tokens=None` that means at most $\lceil n / \texttt{max\_batch} \rceil$ calls.
  The cases are built so that arrival-order batching exceeds one of the two limits;
  one call per pair pads nothing and exceeds the call limit.

## Why this matters

Retrieval pipelines put a cross-encoder behind the vector search: the search
returns a hundred candidates cheaply, and the reranker reads each of them next to the
query to pick the five that go into the prompt. That is a hundred forward passes of
a transformer per user request, so it is batched, and candidates are passages of
anything from one sentence to a page. A batch that mixes a 20-token passage with a
500-token one computes 480 positions of padding for the short one. Sorting by length
first removes most of that waste; a token budget keeps the long batches from running
out of memory while letting short pairs travel sixty to a batch. And the bug this
invites is silent: scores handed back in sorted order look like perfectly good
scores, attached to the wrong documents.
