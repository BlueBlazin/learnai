import torch


def build_mask(lengths: torch.Tensor, T: int, padding_side: str) -> torch.Tensor:
    """Additive attention mask that combines the causal rule with key padding.

    lengths: integer tensor, shape (B,). Row b holds lengths[b] real tokens, 1 <= lengths[b] <= T.
    T: the padded sequence length, a positive int.
    padding_side: "right" (real tokens first) or "left" (real tokens last).
    Returns float32, shape (B, 1, T, T): 0.0 where query i may attend to key j, -inf
    elsewhere. Query i may attend to key j when j <= i and key j is a real token, and
    always when j == i.
    """
    raise NotImplementedError
