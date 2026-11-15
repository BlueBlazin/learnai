## The idea

A transformer processes a batch as a rectangle, so the cost of a batch is its number
of rows times its longest row. The total over all batches is

$$
\sum_{\text{batches } b} |b| \cdot \max_{i \in b} L_i ,
$$

and the only freedom is which pairs share a batch. Every pair pays for the longest
pair of its batch, so a pair wants neighbors of its own length. Sorting by length
and cutting the sorted list into consecutive runs achieves that: within a run the
lengths differ as little as the data allows. (An exchange argument shows that some
optimal partition is always made of consecutive runs of the sorted order: if a
short pair sits in a batch with a longer maximum than a longer pair's batch,
swapping the two never increases either maximum.)

Where to cut is the second decision. A fixed number of pairs per batch is the usual
answer, and it wastes memory in the opposite way: a batch size that is safe for
the longest pairs is far too small for the short ones. A **token budget** bounds
what actually costs memory and time, the area of the rectangle. With ascending
order the pair being added is the longest of its batch, so the test is one
multiplication: `(m + 1) * L_p <= max_tokens`.

The third part is the one that goes wrong in practice. The scorer returns scores
in the order of the batch, and the batch is in sorted order. Carry the original
index of every pair through the whole computation and write each score to
`scores[original_index]`. Sorting indices, and never the pairs, makes that the
natural thing to do.

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

With no documents the loop bodies never run and the scorer is never called, which is
what the specification asks for: an empty batch is an error for the scorer.

## Common mistakes

- **Batching in arrival order.** The result is correct and the cost is not: every
  batch that mixes a short and a long passage pads the short one to the long one.
  In the worked example that is 40 padded tokens instead of 28.
- **Returning the scores in sorted order.** Collecting `scores.extend(scorer(batch))`
  gives a list of the right length with the right values in it, in length order.
  If every document has the same length the bug is invisible, which is why a test
  with uniform lengths does not find it.
- **Truncating the pair instead of the document.** Concatenating query and document
  and keeping the last `max_length` tokens cuts the query. The scorer then answers
  a different question. The query is the part that must survive; real rerankers
  truncate with the "only second" strategy for this reason.
- **Measuring before truncating.** Sorting by the untruncated length puts every
  over-long document at the end even though, after truncation, they are all exactly
  `max_length` long. Harmless for the scores, but the budget test then uses the
  wrong width.
- **Ignoring the token budget**, or testing it with the *previous* longest pair
  (`(m + 1) * current_width`) instead of the pair being added, which lets a long
  pair into a batch of short ones and blows the rectangle up after the fact.
- **One call per pair.** No padding at all and the worst possible throughput: the
  point of batching is that a call has a fixed overhead and a GPU is idle when it
  is fed one row at a time.
- **Reversing the ties.** `sorted(..., key=score, reverse=True)` keeps equal
  elements in their original order too, but `sorted(...)[::-1]` or sorting
  `(score, index)` tuples in reverse ranks equal scores by the higher index.
  Duplicate passages are common in a candidate list, so ties are not a corner case.
- **Calling the scorer with an empty batch** when there are no candidates.

## Complexity

Sorting is $O(n \log n)$; batching, scattering the scores, and ranking are linear
and $O(n \log n)$. All of it is negligible next to the model, whose cost is the
padded-token count the problem measures.

## In the wild

- Sentence Transformers sorts inputs by length before batching in
  `SentenceTransformer.encode` (longest first, so that the largest batch comes
  first and an out-of-memory error shows up immediately) and restores the original
  order with the inverse permutation afterwards. Sorting ascending, as here, makes
  the budget test simpler; the padding saved is the same.
- Text Embeddings Inference, Hugging Face's server for embedding and reranking
  models, forms batches under a token budget (`--max-batch-tokens`) instead of a
  fixed batch size, for the reason given above.
- Tokenizers implement "truncate the document, not the query" as
  `truncation="only_second"` for a pair of texts.
