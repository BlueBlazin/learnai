## The idea

A transformer processes a batch as a dense rectangle, so the cost of a batch $b$ is its number of pairs $|b|$ multiplied by the length of its longest pair. Summing over all batches gives the total padded tokens:

$$
\sum_{\text{batches } b} |b| \cdot \max_{i \in b} L_i
$$

where $L_i$ is the truncated length of pair $i$. Every pair pays for the longest pair in its batch, so pairs should be grouped with neighbors of similar length. Sorting by length and slicing into consecutive runs minimizes the length gap within each batch. An exchange argument confirms that an optimal partition always consists of consecutive runs in sorted order. If a short pair sits in a batch with a larger maximum than a longer pair's batch, swapping the two never increases either batch maximum.

Deciding where to cut each batch is the second piece. A fixed batch size wastes memory in the opposite direction, because a batch size small enough for the longest pairs underutilizes hardware on short pairs. A **token budget** bounds the actual rectangle area that drives memory and compute. Because the pairs are sorted in ascending order, a new pair $p$ of length $L_p$ is the longest in its batch so far. If the current batch holds $m$ pairs, checking the budget takes one multiplication: `(m + 1) * L_p <= max_tokens`.

The final piece is restoring the original order. The scorer returns scores in batch order, which follows the length-sorted order. Carrying each pair's original index through sorting and batching lets you write every score directly to `scores[original_index]`. Sorting the indices rather than the pairs themselves makes this straightforward.

## Reference solution (Python)

```python
def rerank(query, docs, scorer, max_batch, top_n, max_tokens=None):
    room = scorer.max_length - len(query)
    pairs = [(query, doc[:room]) for doc in docs]
    lengths = [len(query) + len(doc) for _, doc in pairs]
    order = sorted(range(len(pairs)), key=lambda i: lengths[i])

    batches = []
    for i in order:
        batch = batches[-1] if batches else None
        full = batch is None or len(batch) == max_batch
        if not full and max_tokens is not None:
            full = (len(batch) + 1) * lengths[i] > max_tokens
        if full:
            batches.append([i])
        else:
            batch.append(i)

    scores = [0.0] * len(pairs)
    for batch in batches:
        for i, score in zip(batch, scorer([pairs[i] for i in batch]), strict=True):
            scores[i] = score

    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(top_n, 0)]
    return scores, ranking
```

When `docs` is empty, `batches` is empty and the scoring loop never runs, so `scorer` is never called with an empty batch.

## Common mistakes

- **Batching in arrival order.** The scores and ranking are right, but the padded-token cost is too high whenever a short passage shares a batch with a long one. In the worked example, arrival order uses 40 padded tokens instead of 28.
- **Returning scores in sorted order.** Appending results with `scores.extend(scorer(batch))` returns a list of the right length with the right floats, but in length order rather than document order. Tests where all documents have the same length will not catch this bug.
- **Truncating the pair instead of the document.** Concatenating query and document and keeping the last `max_length` tokens cuts off the query, causing the scorer to evaluate a different question. Real rerankers truncate only the second sequence for this reason.
- **Measuring length before truncating.** Sorting by untruncated length puts over-long documents at the end even though they all truncate to `max_length`, and it causes the token-budget check to use the wrong width.
- **Ignoring the token budget or checking the wrong width.** Testing `(m + 1) * current_width` against `max_tokens` instead of `(m + 1) * L_p` lets a long pair join a batch of short pairs and overflow the rectangle.
- **Scoring one pair per call.** This eliminates padding completely, but it maximizes call overhead and leaves GPUs underutilized, failing the call-count limit.
- **Breaking score ties in reverse.** Python's `sorted(..., key=..., reverse=True)` is stable and preserves index order for ties, but reversing a sorted list with `[::-1]` or sorting `(score, index)` tuples in reverse puts higher indices first. Duplicate passages are common in retrieval candidates, so ties happen often.
- **Calling `scorer` on an empty batch.** When `docs` is empty, `scorer([])` raises `ValueError`.

## Complexity

Let $n = \text{len}(\texttt{docs})$. Sorting by length and ranking by score each take $O(n \log n)$ time. Truncating, batching, and scattering scores back to their original indices are $O(n)$. All of this overhead is negligible compared to the cross-encoder forward passes, whose cost is measured by `padded_tokens`.

## In the wild

- Sentence Transformers sorts inputs by length inside `SentenceTransformer.encode` (longest first, so any out-of-memory failure surfaces immediately on the first batch) and restores the original order with an inverse permutation afterward. Sorting ascending gives the same padding savings while keeping the token-budget check to one multiplication.
- Hugging Face's Text Embeddings Inference (TEI) server packs embedding and reranking requests under a token budget (`--max-batch-tokens`) rather than a fixed batch size.
- Hugging Face tokenizers implement "truncate the document, not the query" via `truncation="only_second"` on text pairs.
