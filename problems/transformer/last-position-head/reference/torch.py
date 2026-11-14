import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    # Select first, project second: only the last position predicts the next token,
    # so the (V, D) matrix is applied to B vectors instead of B * T.
    last = hidden[:, -1]  # (B, D)
    return last @ w_out.T  # (B, V); w_out is (out, in)
