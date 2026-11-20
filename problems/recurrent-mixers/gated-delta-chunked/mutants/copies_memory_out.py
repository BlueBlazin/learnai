# BUG: reads the whole memory out with identity rows, loops over tokens on the copy, and writes the result back: Dk rows in one call, more than max_rows when keys are wide.
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
    d_k = q.shape[-1]
    eye = torch.eye(d_k).expand(q.shape[1], d_k, d_k)
    S = memory.read(eye).transpose(-1, -2)  # I @ S^T is the memory itself
    g, ys = torch.exp(log_g), []
    for t in range(q.shape[0]):
        S = S * g[t, :, None, None]
        delta = beta[t, :, None] * (v[t] - torch.einsum("hvk,hk->hv", S, k[t]))
        S = S + delta[:, :, None] * k[t, :, None, :]
        ys.append(torch.einsum("hvk,hk->hv", S, q[t]))
    # decay 0 and identity keys replace the memory by S.
    return torch.stack(ys), memory.update(torch.zeros(q.shape[1]), S.transpose(-1, -2), eye)
