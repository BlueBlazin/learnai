# BUG: max_tokens is ignored: batches are cut by max_batch alone, so a batch of long pairs exceeds the token budget.
def rerank(
    query: list[int],
    docs: list[list[int]],
    scorer,
    max_batch: int,
    top_n: int,
    max_tokens: int | None = None,
) -> tuple[list[float], list[int]]:
    room = scorer.max_length - len(query)
    pairs = [(query, doc[:room]) for doc in docs]
    lengths = [len(query) + len(doc) for _, doc in pairs]

    order = sorted(range(len(pairs)), key=lambda i: lengths[i])

    batches: list[list[int]] = []
    for i in order:
        batch = batches[-1] if batches else None
        full = batch is None or len(batch) == max_batch
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
