# BUG: every pair is scored in a call of its own: nothing is padded, and the number of calls is the number of documents.
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
        full = True
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
