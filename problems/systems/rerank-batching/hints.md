The scorer charges every pair in a batch for the length of the longest pair in that batch. Think about how to group pairs so that the longest pair in each batch is close in length to the rest. Once you reorder pairs for batching, how will you map each returned score back to its original document?

---

Sort the document *indices* by truncated pair length using `sorted(range(len(docs)), key=lambda i: lengths[i])`, which preserves original order on ties. Build each batch as a list of indices rather than pairs. A new pair joins the current batch unless the batch already has `max_batch` pairs or `(len(batch) + 1) * lengths[i]` exceeds `max_tokens`. Because the order is ascending, `lengths[i]` is always the new maximum length of the batch. Be sure to truncate each document from the end before measuring its length, and never truncate the query.

---

```python
scores = [0.0] * len(docs)
for batch in batches:  # each batch is a list of document indices
    results = scorer([pairs[i] for i in batch])
    for i, score in zip(batch, results):
        scores[i] = score  # place score back at the document's original index
ranking = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
```

Because `sorted` is stable, sorting indices by `-scores[i]` keeps equal scores in ascending index order. Reversing a sorted list or sorting `(score, index)` in descending order would reverse ties.
