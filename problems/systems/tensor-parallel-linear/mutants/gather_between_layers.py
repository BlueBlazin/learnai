# BUG: the hidden activations are all-gathered between the two layers, a second collective on the widest tensor of the MLP, before each device takes its slice back and continues.
from itertools import pairwise

import torch
import torch.nn.functional as F


def shard_weights(
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    n: int,
) -> list[dict[str, torch.Tensor]]:
    h = w_up.shape[0]
    bounds = [i * h // n for i in range(n + 1)]
    return [
        {
            "up": w_up[lo:hi],
            "up_bias": b_up[lo:hi],
            "down": w_down[:, lo:hi],
            "down_bias": b_down,
        }
        for lo, hi in pairwise(bounds)
    ]


def tp_mlp(x: torch.Tensor, devs) -> torch.Tensor:
    n = devs.n_devices
    hidden = devs.all_gather(
        [F.silu(x @ devs.get(i, "up").T + devs.get(i, "up_bias")) for i in range(n)]
    )
    parts, lo = [], 0
    for i in range(n):
        down = devs.get(i, "down")
        hi = lo + down.shape[1]
        parts.append(hidden[..., lo:hi] @ down.T)
        lo = hi
    return devs.all_reduce(parts) + devs.get(0, "down_bias")
