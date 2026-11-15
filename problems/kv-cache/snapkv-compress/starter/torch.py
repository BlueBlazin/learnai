import torch


def snapkv_select(attn: torch.Tensor, window: int, budget: int, kernel: int) -> torch.Tensor:
    """Choose, per head, the prompt positions a compressed KV cache keeps.

    attn: float32, shape (B, H, L, L): the causal attention weights of a prompt of L
        tokens in one layer; attn[b, h, i, j] is the weight of query i on position j.
    window: the last `window` prompt positions are the observation window.
    budget: the number of positions to keep per head (window <= budget).
    kernel: odd width of the max-pooling over the votes (1 means no pooling).
    Returns an integer tensor of shape (B, H, min(L, budget)): the kept positions of
    each head in increasing order, ending with the window's positions.
    """
    raise NotImplementedError
