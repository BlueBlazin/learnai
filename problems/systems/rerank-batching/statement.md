A cross-encoder reranker scores a query and a document together to produce a single relevance value. Implement `rerank` to score a list of candidate documents against a query while minimizing padded tokens. You will truncate long documents, sort pairs by length, pack them into batches under size and token limits, score each batch, and restore the scores to the original document order.

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

**The scorer.** `scorer` is a `CountingScorer` object. A *pair* is a tuple `(query_tokens, doc_tokens)` of two non-empty sequences of integer token IDs. Calling `scorer(pairs)` takes a non-empty list of pairs and returns one float per pair in the same order. Each pair's score depends only on that pair, regardless of what else is in the batch. The scorer raises `ValueError` if `pairs` is empty or if any pair has `len(query_tokens) + len(doc_tokens) > scorer.max_length`.

The scorer processes each batch as a padded rectangle where every pair is padded to the longest pair in that batch. Each call adds the padded rectangle area to `scorer.counters.padded_tokens`:

$$
\text{padded tokens of a call} = (\text{number of pairs}) \times (\text{length of the longest pair})
$$

Each call also increments `scorer.counters.calls` by 1 and adds the batch size to `scorer.counters.pairs`.

**Behavior of `rerank`.** Let $q = \text{len}(\texttt{query})$ and $n = \text{len}(\texttt{docs})$.

1. **Truncate.** For each document `docs[i]`, keep only its first `scorer.max_length - q` tokens if it is longer. Never shorten `query`. Form pair $i$ as `(query, truncated doc i)` with total length $L_i = q + \text{len}(\text{truncated doc } i)$.
2. **Sort.** Order the pairs by $L_i$ from shortest to longest. Preserve the original order for pairs with equal length.
3. **Batch.** Walk through the sorted pairs and build one batch at a time. Let $m$ be the number of pairs already in the current batch, and let $p$ be the next pair with length $L_p$. Because the pairs are sorted in ascending order, $L_p$ is the longest length in the batch if $p$ is added. Add $p$ to the current batch unless doing so violates either limit:
   - $m + 1 > \texttt{max\_batch}$ (too many pairs), or
   - `max_tokens` is not `None` and $(m + 1) \cdot L_p > \texttt{max\_tokens}$ (the padded rectangle exceeds the token budget).

   If adding $p$ would violate a limit, close the current batch and start a new batch containing $p$.
4. **Score.** Call `scorer` once per batch so every pair is scored exactly once.
5. **Restore.** Place each score back at its original document index so that `scores[i]` is the score of `docs[i]`.
6. **Rank.** Build `ranking` as the indices of the `min(top_n, n)` documents with the highest scores, ordered from highest to lowest. Break ties by choosing the lower document index first.

Return `(scores, ranking)`, where `scores` is a list of $n$ floats and `ranking` is a list of ints. If `docs` is empty, return `([], [])` without calling `scorer`. Do not modify `query`, `docs`, or any individual document list.

## Example

```python
query = [7, 3]  # q = 2
docs = [[5], [9, 9, 9, 9, 9, 9], [4, 8], [6, 6, 6, 6, 6, 6], [2]]
#  L =   3          8              4            8             3
scores, ranking = rerank(query, docs, scorer, max_batch=4, top_n=2, max_tokens=12)
```

Sorted by length, the documents are in index order `0, 4, 2, 1, 3` with pair lengths `3, 3, 4, 8, 8`.

| step | pair | $m$ | $(m + 1) \cdot L_p$ | decision |
|---|---|---|---|---|
| 1 | doc 0, $L = 3$ | 0 | 3 | starts batch 1 |
| 2 | doc 4, $L = 3$ | 1 | 6 | joins batch 1 |
| 3 | doc 2, $L = 4$ | 2 | 12 | joins batch 1 (12 does not exceed 12) |
| 4 | doc 1, $L = 8$ | 3 | 32 | exceeds `max_tokens`, starts batch 2 |
| 5 | doc 3, $L = 8$ | 1 | 16 | exceeds `max_tokens`, starts batch 3 |

This produces three calls with $3 \cdot 4 + 1 \cdot 8 + 1 \cdot 8 = 28$ padded tokens. Batching the same documents in arrival order under the same limits produces five single-pair batches (`[0]`, `[1]`, `[2]`, `[3]`, `[4]`) and five calls.

Without a token limit (`max_tokens=None`) and with `max_batch=3`, length-sorted batching forms `[0, 4, 2]` and `[1, 3]`, using 28 padded tokens in two calls. Arrival order forms `[0, 1, 2]` and `[3, 4]`, using $3 \cdot 8 + 2 \cdot 8 = 40$ padded tokens.

Regardless of batching order, `scores[1]` must hold the score of `docs[1]` (the pair scored in batch 2), not the second score returned overall.

## Constraints

- $1 \le q < \texttt{scorer.max\_length}$, so every truncated document retains at least one token.
- No document in `docs` is empty, and $n \le 500$.
- `max_batch` $\ge 1$ and `top_n` $\ge 0$.
- When `max_tokens` is not `None`, `max_tokens` $\ge \texttt{scorer.max\_length}$, so every pair fits in a batch of size 1.
- Each value in `scores` must exactly match the scorer's output for that truncated pair, and `ranking` must match exactly.
- **Cost limits.** `scorer.counters.padded_tokens` and `scorer.counters.calls` must not exceed what the batching rule above produces. `scorer.counters.pairs` must equal $n$, and no call may exceed `max_batch` or `max_tokens`. When `max_tokens` is `None`, the number of calls must be at most $\lceil n / \texttt{max\_batch} \rceil$. Test cases are constructed so that arrival-order batching violates one of these limits, and scoring one pair per call violates the call limit.

## Why this matters

Retrieval pipelines often run a cheap vector search to fetch a hundred candidate passages, then pass those candidates through a cross-encoder reranker to pick the top five for an LLM prompt. Because passages vary from a single sentence to a full page, mixing a 20-token passage with a 500-token passage in one batch forces the model to compute 480 wasted padding positions.

Sorting by length groups similar passages together, and a token budget prevents long batches from running out of GPU memory while allowing short pairs to pack densely. Keeping track of original indices is just as critical, because returning scores in sorted order is a silent bug that attaches valid-looking scores to the wrong documents.
