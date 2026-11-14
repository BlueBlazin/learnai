import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Divide each row of logits by its own temperature; temperature 0 means greedy.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    temperature: float32 tensor of shape (B,), one temperature per row, each >= 0.
    Returns a float32 tensor of shape (B, V). A row with temperature t > 0 is
    logits / t. A row with temperature exactly 0 is 0 at its largest logit (the
    lowest index among ties) and -inf everywhere else.
    """
    raise NotImplementedError
