# BUG: the expert device sends back raw expert outputs, one row per (token, expert), and the home device applies the weights: the results are not summed where they were computed, so a pair's result message is larger than its row message whenever two of a token's experts share that device, and never carries the weighted partial result.
import numpy as np
import torch

from whetstone.kit import CountingExperts, SimDevices


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
            tokens = np.flatnonzero((homes == src) & (target == dst).any(axis=1))
            if tokens.size == 0:
                continue
            rows = x[torch.from_numpy(tokens)]
            batches.append((src, tokens, rows if src == dst else devs.send(src, dst, rows)))
        if not batches:
            continue
        rows = torch.cat([batch[2] for batch in batches])
        tokens = np.concatenate([batch[1] for batch in batches])
        owner = np.concatenate([np.full(batch[1].size, batch[0]) for batch in batches])

        # The expert device returns raw expert outputs, one row per (token, expert).
        raw = {src: [] for src, _, _ in batches}
        for expert in np.flatnonzero(placed == dst).tolist():
            at, slot = np.nonzero(routes[tokens] == expert)
            if at.size == 0:
                continue
            out = experts(expert, rows[torch.from_numpy(at)])
            for src, parts in raw.items():
                keep = np.flatnonzero(owner[at] == src)
                if keep.size:
                    parts.append((tokens[at[keep]], slot[keep], out[torch.from_numpy(keep)]))

        # The home device applies the weights.
        for src, parts in raw.items():
            back = torch.cat([part[2] for part in parts])
            if src != dst:
                back = devs.send(dst, src, back)
            sent = torch.from_numpy(np.concatenate([part[0] for part in parts]))
            slot = torch.from_numpy(np.concatenate([part[1] for part in parts]))
            y = y.index_add(0, sent, weights[sent, slot][:, None] * back)
    return y
