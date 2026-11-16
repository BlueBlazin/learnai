# BUG: runs an expert once for each device that sent it rows instead of once on all of its rows: the same rows in more, smaller calls.
import numpy as np
import torch

from learnai.kit import CountingExperts, SimDevices


def ep_forward(
    x: torch.Tensor,
    home: list[int],
    w_router: torch.Tensor,
    experts: CountingExperts,
    placement: list[int],
    devs: SimDevices,
    k: int,
) -> torch.Tensor:
    # Route as in moe-sparse-dispatch (the sort is stable: ties go to the lowest index).
    logits = x @ w_router.T  # (N, E)
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]  # (N, k)
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)  # (N, k)

    # Bookkeeping on the host: which device each of a token's experts lives on.
    routes = chosen.numpy()
    homes, placed = np.asarray(home), np.asarray(placement)
    target = placed[routes]  # (N, k)

    y = torch.zeros_like(x)
    for dst in range(devs.n_devices):  # the device whose experts work in this round
        # Dispatch: from every device, the rows of its tokens that need an expert on
        # dst. A token appears once, however many of its experts live there.
        batches = []
        for src in range(devs.n_devices):
            tokens = np.flatnonzero((homes == src) & (target == dst).any(axis=1))
            if tokens.size == 0:
                continue  # nothing to send: no message
            rows = x[torch.from_numpy(tokens)]
            batches.append((src, tokens, rows if src == dst else devs.send(src, dst, rows)))
        if not batches:
            continue
        rows = torch.cat([batch[2] for batch in batches])  # everything now on dst
        tokens = np.concatenate([batch[1] for batch in batches])  # the token of each row

        # Compute on dst: each of its experts once, on its rows from every source,
        # weighted and summed per token.
        partial = torch.zeros_like(rows)
        for expert in np.flatnonzero(placed == dst).tolist():
            at, slot = np.nonzero(routes[tokens] == expert)  # rows on dst, and their top-k slot
            origin = homes[tokens[at]]
            for src in np.unique(origin).tolist():
                part_at, part_slot = at[origin == src], slot[origin == src]
                weight = weights[torch.from_numpy(tokens[part_at]), torch.from_numpy(part_slot)]
                part_at = torch.from_numpy(part_at)
                out = experts(expert, rows[part_at])
                partial = partial.index_add(0, part_at, weight[:, None] * out)

        # Combine: one message back to every source, a row per row it sent.
        start = 0
        for src, sent, _ in batches:
            part = partial[start : start + sent.size]
            start += sent.size
            if src != dst:
                part = devs.send(dst, src, part)
            y = y.index_add(0, torch.from_numpy(sent), part)
    return y
