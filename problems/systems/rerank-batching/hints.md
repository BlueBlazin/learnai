The scorer charges every pair of a batch for the longest pair in it. Which pairs
should share a batch so that the longest one is not much longer than the rest? And
once you have reordered the pairs to arrange that, how will you know which document
each returned score belongs to?

---

Do not sort the pairs themselves: sort their *indices* by pair length
(`sorted(range(n), key=lambda i: lengths[i])`, which is stable). Build the batches
as lists of indices. A new pair joins the last batch unless the batch already has
`max_batch` pairs or `(len(batch) + 1) * lengths[i]` would exceed `max_tokens`; the
new pair is the longest, because the order is ascending. Truncate documents before
you measure them, and take the tokens away from the document's end, never from the
query.

---

```python
scores = [0.0] * len(docs)
for batch in batches:  # each batch is a list of document indices
    results = scorer([pairs[i] for i in batch])
    for i, score in zip(batch, results):
        scores[i] = score  # back to the document's own slot
ranking = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
```

`sorted` is stable, so sorting by the negated score keeps equal scores in index
order. Sorting with `reverse=True` would also reverse the ties.
