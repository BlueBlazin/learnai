# BUG: the output bias is added to every device's partial output before the all-reduce, so the result contains it n times.
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
    parts = []
    for i in range(devs.n_devices):
        hidden = F.silu(x @ devs.get(i, "up").T + devs.get(i, "up_bias"))
        parts.append(hidden @ devs.get(i, "down").T + devs.get(i, "down_bias"))
    return devs.all_reduce(parts)
