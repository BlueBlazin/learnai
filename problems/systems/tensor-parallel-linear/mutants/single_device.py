# BUG: every shard is fetched and concatenated and the unsplit layer is computed in one place, with no communication at all: nothing is parallel.
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
    w_up = torch.cat([devs.get(i, "up") for i in range(n)], dim=0)
    b_up = torch.cat([devs.get(i, "up_bias") for i in range(n)], dim=0)
    w_down = torch.cat([devs.get(i, "down") for i in range(n)], dim=1)
    return F.silu(x @ w_up.T + b_up) @ w_down.T + devs.get(0, "down_bias")
