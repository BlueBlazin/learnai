import mlx.core as mx


def index_scores(q_idx: mx.array, k_idx: mx.array, w: mx.array) -> mx.array:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    # One dot product per (query, head, key); the ReLU acts on each head separately.
    per_head = mx.maximum(mx.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5, 0.0)
    # The head weights keep their sign: a head can lower a score.
    return mx.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: mx.array, visible: mx.array, k: int) -> mx.array:
    # Invisible entries go last, whatever their score.
    masked = mx.where(visible, scores, -mx.inf)
    # mx.argsort is stable: ascending in -masked ranks equal scores by index, lowest first.
    order = mx.argsort(-masked, axis=-1)
    rank = mx.argsort(order, axis=-1)  # rank[t, s]: the place of entry s in row t
    # Rows with fewer than k visible entries rank some invisible ones below k.
    return (rank < k) & visible
