import torch


def hc_weights(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The weights of one hyper-connection, and the sublayer input they produce.

    streams: float32, shape (T, N, D). The N residual streams of T tokens.
    fn: float32, shape (2 * N + N * N, N * D). base: float32, shape (2 * N + N * N,).
    scale: float32, shape (3,). One factor each for pre, post, and comb.
    iters: Sinkhorn iterations, >= 1. eps: a small positive float.
    Returns (collapsed, post, comb): float32 of shapes (T, D), (T, N), (T, N, N).
    """
    raise NotImplementedError


def hc_merge(
    streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Write a sublayer's output back to the N streams.

    streams: float32, shape (T, N, D). The streams the weights were computed from.
    y: float32, shape (T, D). The sublayer's output.
    post: float32, shape (T, N). comb: float32, shape (T, N, N).
    Returns float32, shape (T, N, D).
    """
    raise NotImplementedError
