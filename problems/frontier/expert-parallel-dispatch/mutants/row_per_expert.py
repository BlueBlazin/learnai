# BUG: dispatches one row per (token, expert) pair instead of one row per token and destination device: a token whose two experts live on the same remote device crosses twice, and two result rows come back for it.
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
    for dst in range(devs.n_devices):
        batches = []
        for src in range(devs.n_devices):
            # One row per (token, expert) pair: a token with two experts on dst is in twice.
            tokens, slots = np.nonzero((homes == src)[:, None] & (target == dst))
            if tokens.size == 0:
                continue
            rows = x[torch.from_numpy(tokens)]
            if src != dst:
                rows = devs.send(src, dst, rows)
            batches.append((src, tokens, slots, rows))
        if not batches:
            continue
        rows = torch.cat([batch[3] for batch in batches])
        tokens = np.concatenate([batch[1] for batch in batches])
        slots = np.concatenate([batch[2] for batch in batches])

        partial = torch.zeros_like(rows)
        for expert in np.flatnonzero(placed == dst).tolist():
            at = np.flatnonzero(routes[tokens, slots] == expert)
            if at.size == 0:
                continue
            weight = weights[torch.from_numpy(tokens[at]), torch.from_numpy(slots[at])]
            at = torch.from_numpy(at)
            partial = partial.index_add(0, at, weight[:, None] * experts(expert, rows[at]))

        start = 0
        for src, sent, _, _ in batches:
            part = partial[start : start + sent.size]
            start += sent.size
            if src != dst:
                part = devs.send(dst, src, part)
            y = y.index_add(0, torch.from_numpy(sent), part)
    return y
