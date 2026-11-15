def rerank(
    query: list[int],
    docs: list[list[int]],
    scorer,
    max_batch: int,
    top_n: int,
    max_tokens: int | None = None,
) -> tuple[list[float], list[int]]:
    # Truncate the document, never the query: without the query the score means nothing.
    room = scorer.max_length - len(query)
    pairs = [(query, doc[:room]) for doc in docs]
    lengths = [len(query) + len(doc) for _, doc in pairs]

    # Shortest pairs first; sorted() is stable, so equal lengths stay in arrival order.
    order = sorted(range(len(pairs)), key=lambda i: lengths[i])

    # Fill a batch until one more pair would break a limit. The pair being added is
    # the longest so far, so it sets the width of the rectangle the scorer pays for.
    batches: list[list[int]] = []
    for i in order:
        batch = batches[-1] if batches else None
        full = batch is None or len(batch) == max_batch
        if not full and max_tokens is not None:
            full = (len(batch) + 1) * lengths[i] > max_tokens
        if full:
            batches.append([i])
        else:
            batch.append(i)

    # Each score goes back to the slot of the document it belongs to.
    scores = [0.0] * len(pairs)
    for batch in batches:
        for i, score in zip(batch, scorer([pairs[i] for i in batch]), strict=True):
            scores[i] = score

    # Best first; sorted() is stable, so equal scores keep the lower index first.
    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(top_n, 0)]
    return scores, ranking
