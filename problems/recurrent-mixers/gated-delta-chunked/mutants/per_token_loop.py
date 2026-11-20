# BUG: correct numbers, but sequential over tokens: one read and one write of the memory per token instead of per chunk.
from typing import Any

import torch


def gdn_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    memory: Any,
    chunk_size: int,
) -> tuple[torch.Tensor, Any]:
    g, ys = torch.exp(log_g), []
    for t in range(q.shape[0]):
        # One call returns S k and S q; the decay multiplies both.
        seen = g[t, :, None, None] * memory.read(torch.stack((k[t], q[t]), dim=1))
        delta = beta[t, :, None] * (v[t] - seen[:, 0])
        ys.append(seen[:, 1] + delta * (k[t] * q[t]).sum(-1, keepdim=True))
        memory = memory.update(g[t], delta[:, None, :], k[t][:, None, :])
    return torch.stack(ys), memory
