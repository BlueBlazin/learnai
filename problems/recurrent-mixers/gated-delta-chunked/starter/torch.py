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
    """The gated delta rule over T tokens, sequential over chunks only.

        q, k: float32, shape (T, H, Dk). v: float32, shape (T, H, Dv).
        log_g: float32, shape (T, H), <= 0. The log of the decay of each token and head.
        beta: float32, shape (T, H), in [0, 1].
    memory: the memory before the first token, an object (see the statement):
            memory.read(rows)                   rows (H, R, Dk) -> rows @ S^T, (H, R, Dv)
            memory.update(decay, values, keys)  -> a new memory, decay * S + values^T keys
            memory.max_rows                     the most rows one call accepts
        chunk_size: int >= 1; T need not be a multiple of it.
        Returns (Y, memory): Y float32 of shape (T, H, Dv), the read of every token, and
        the memory after the last token, as returned by the last update. Both equal what
        T calls of the one-token rule with g = exp(log_g) give.
    """
    raise NotImplementedError
