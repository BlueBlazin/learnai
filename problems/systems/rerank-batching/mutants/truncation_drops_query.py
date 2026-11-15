# BUG: an over-long pair is cut from the front (the last max_length tokens of query + document are kept), which removes query tokens instead of document tokens.
def rerank(
    query: list[int],
    docs: list[list[int]],
    scorer,
    max_batch: int,
    top_n: int,
    max_tokens: int | None = None,
) -> tuple[list[float], list[int]]:
    pairs = []
    for doc in docs:
        excess = max(len(query) + len(doc) - scorer.max_length, 0)
        cut = min(excess, len(query) - 1)
        pairs.append((query[cut:], doc[: len(doc) - (excess - cut)]))
    lengths = [len(q) + len(d) for q, d in pairs]

    order = sorted(range(len(pairs)), key=lambda i: lengths[i])

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

    scores = [0.0] * len(pairs)
    for batch in batches:
        for i, score in zip(batch, scorer([pairs[i] for i in batch]), strict=True):
            scores[i] = score

    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(top_n, 0)]
    return scores, ranking
