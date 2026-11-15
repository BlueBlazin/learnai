def rerank(
    query: list[int],
    docs: list[list[int]],
    scorer,
    max_batch: int,
    top_n: int,
    max_tokens: int | None = None,
) -> tuple[list[float], list[int]]:
    """Score every (query, document) pair with a cross-encoder, in length-sorted batches.

    query: the query's token ids (not empty).
    docs: one list of token ids per document (none empty); may be an empty list.
    scorer: a kit CountingScorer. scorer(pairs) scores a list of (query, doc) pairs
        and returns one float per pair; scorer.max_length is the longest pair it takes.
    max_batch: the most pairs one call may hold.
    top_n: how many documents to rank.
    max_tokens: the most padded tokens (pairs times the longest pair) one call may
        hold, or None for no such limit.

    Returns (scores, ranking): scores[i] is the score of docs[i]; ranking holds the
    indices of the min(top_n, len(docs)) best documents, best first, ties to the
    lower index.
    """
    raise NotImplementedError
