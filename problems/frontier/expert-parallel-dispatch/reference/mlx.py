import mlx.core as mx
import numpy as np

from learnai.kit import CountingExperts, SimDevices


def ep_forward(
    x: mx.array,
    home: list[int],
    w_router: mx.array,
    experts: CountingExperts,
    placement: list[int],
    devs: SimDevices,
    k: int,
) -> mx.array:
    # Route as in moe-sparse-dispatch (the sort is stable: ties go to the lowest index).
    logits = x @ w_router.T  # (N, E)
    chosen = mx.argsort(-logits, axis=-1)[:, :k]  # (N, k)
    weights = mx.softmax(mx.take_along_axis(logits, chosen, axis=-1), axis=-1)  # (N, k)

    # Bookkeeping on the host: which device each of a token's experts lives on.
    routes = np.array(chosen)
    homes, placed = np.asarray(home), np.asarray(placement)
    target = placed[routes]  # (N, k)

    y = mx.zeros_like(x)
    for dst in range(devs.n_devices):  # the device whose experts work in this round
        # Dispatch: from every device, the rows of its tokens that need an expert on
        # dst. A token appears once, however many of its experts live there.
        batches = []
        for src in range(devs.n_devices):
            tokens = np.flatnonzero((homes == src) & (target == dst).any(axis=1))
            if tokens.size == 0:
                continue  # nothing to send: no message
            rows = x[mx.array(tokens)]
            batches.append((src, tokens, rows if src == dst else devs.send(src, dst, rows)))
        if not batches:
            continue
        rows = mx.concatenate([batch[2] for batch in batches])  # everything now on dst
        tokens = np.concatenate([batch[1] for batch in batches])  # the token of each row

        # Compute on dst: each of its experts once, on its rows from every source,
        # weighted and summed per token.
        partial = mx.zeros_like(rows)
        for expert in np.flatnonzero(placed == dst).tolist():
            at, slot = np.nonzero(routes[tokens] == expert)  # rows on dst, and their top-k slot
            if at.size == 0:
                continue
            weight = weights[mx.array(tokens[at]), mx.array(slot)]
            at = mx.array(at)
            partial = partial.at[at].add(weight[:, None] * experts(expert, rows[at]))

        # Combine: one message back to every source, a row per row it sent.
        start = 0
        for src, sent, _ in batches:
            part = partial[start : start + sent.size]
            start += sent.size
            if src != dst:
                part = devs.send(dst, src, part)
            y = y.at[mx.array(sent)].add(part)
    return y
