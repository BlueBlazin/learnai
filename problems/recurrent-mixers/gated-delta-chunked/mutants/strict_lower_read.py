# BUG: the within-chunk read excludes the diagonal, so a token does not see its own write.
from typing import Any

import torch
import torch.nn.functional as F


def gdn_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    memory: Any,
    chunk_size: int,
) -> tuple[torch.Tensor, Any]:
    steps, heads = q.shape[0], q.shape[1]
    pad = -steps % chunk_size

    def chunks(x: torch.Tensor) -> torch.Tensor:
        # (T, H, ...) -> (H, chunks, C, ...). Zero rows fill the last chunk: a zero key
        # and value write nothing, and a log-decay of 0 leaves the memory as it is.
        x = F.pad(x.transpose(0, 1), (0, 0) * (x.ndim - 2) + (0, pad))
        return x.reshape(heads, -1, chunk_size, *x.shape[2:])

    q, k, v, log_g, beta = (chunks(x) for x in (q, k, v, log_g, beta))

    # Decay between two positions of a chunk: G[i, j] = exp(c_i - c_j) for i >= j.
    # The upper triangle would be exp of a positive number: mask it before the exp.
    c = torch.cumsum(log_g, dim=-1)  # (H, chunks, C)
    lower = torch.ones(chunk_size, chunk_size, dtype=torch.bool).tril()
    G = torch.exp((c[..., :, None] - c[..., None, :]).masked_fill(~lower, -torch.inf))

    # Everything that does not depend on the incoming memory, for all chunks at once.
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    system = torch.eye(chunk_size) + ((k_beta @ k.transpose(-1, -2)) * G).tril(-1)
    U = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(
        system, k_beta * torch.exp(c)[..., None], upper=False, unitriangular=True
    )
    read = ((q @ k.transpose(-1, -2)) * G).tril(-1)  # strictly earlier tokens only
    q_start = q * torch.exp(c)[..., None]  # queries reading the memory at chunk start
    k_end = k * torch.exp(c[..., -1:] - c)[..., None]  # keys decayed to the chunk end
    total = torch.exp(c[..., -1])  # decay of the whole chunk, (H, chunks)

    # The only sequential part: per chunk, one read of the memory and one write.
    ys = []
    for i in range(q.shape[1]):
        # Rows of W and of the decayed queries, read against the memory in one call.
        seen = memory.read(torch.cat((W[:, i], q_start[:, i]), dim=1))  # (H, 2C, Dv)
        written = U[:, i] - seen[:, :chunk_size]  # what each token really writes
        ys.append(seen[:, chunk_size:] + read[:, i] @ written)
        memory = memory.update(total[:, i], written, k_end[:, i])
    Y = torch.cat(ys, dim=1)[:, :steps]  # drop the padding
    return Y.transpose(0, 1), memory
