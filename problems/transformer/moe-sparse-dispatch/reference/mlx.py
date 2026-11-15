import mlx.core as mx
import numpy as np

from whetstone.kit import CountingExperts


def moe_sparse(x: mx.array, w_router: mx.array, experts: CountingExperts, k: int) -> mx.array:
    n_tokens = x.shape[0]
    # Route: the k best experts of every token, best first. The sort is stable, so
    # equal logits are ranked by expert index, lowest first.
    logits = x @ w_router.T  # (N, E)
    chosen = mx.argsort(-logits, axis=-1)[:, :k]  # (N, k); stable
    # A softmax over the kept logits is the full softmax renormalized over them.
    weights = mx.softmax(mx.take_along_axis(logits, chosen, axis=-1), axis=-1)  # (N, k)

    # Flatten to N * k (token, expert, weight) assignments and sort them by expert:
    # each expert's assignments become one contiguous run.
    flat_expert = chosen.reshape(-1)
    flat_token = mx.repeat(mx.arange(n_tokens), k)
    flat_weight = weights.reshape(-1)
    order = mx.argsort(flat_expert)  # stable
    # The run lengths decide the shapes of the calls, so they are read on the host.
    counts = np.bincount(np.array(flat_expert), minlength=experts.n_experts).tolist()

    y = mx.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue  # nobody chose this expert: no call at all
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]  # the rows this expert must process, in any order
        out = experts(expert, x[tokens])  # one call, exactly its rows
        # Scatter back: row j of `out` belongs to token tokens[j]. A token never picks
        # the same expert twice, so the indices of one run are distinct.
        y = y.at[tokens].add(flat_weight[run, None] * out)
    return y
