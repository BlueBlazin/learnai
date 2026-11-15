import mlx.core as mx


def search(
    queries: mx.array, corpus: mx.array, k: int, metric: str = "cosine"
) -> tuple[mx.array, mx.array]:
    if metric == "cosine":
        # Cosine similarity is the inner product of unit vectors: normalize both sides.
        queries = queries / mx.linalg.norm(queries, axis=1, keepdims=True)
        corpus = corpus / mx.linalg.norm(corpus, axis=1, keepdims=True)
    scores = queries @ corpus.T  # (Q, N): every query against every vector, one product
    k = min(k, corpus.shape[0])

    # mx.topk returns values only and mx.argpartition leaves ties in an undefined
    # order. mx.argsort is stable and runs on the GPU, where sorting a whole row is
    # as fast as partitioning it: sort the negated scores (descending, lower index
    # first among equal scores) and keep the first k columns.
    index = mx.argsort(-scores, axis=1)[:, :k]
    return index.astype(mx.int32), mx.take_along_axis(scores, index, axis=1)
