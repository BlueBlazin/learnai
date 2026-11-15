import torch


def forward_amp(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` in mixed precision and compute sampling statistics in float32.

    model: a module with float32 parameters on `device`; model(x) returns logits (B, V).
    x: float32 tensor of shape (B, D), on `device`.
    device: the device of the model and of `x`.
    dtype: the 16-bit type of the mixed-precision region, torch.float16 or torch.bfloat16.
    temperature: a positive float; the logits are divided by it in float32.
    Returns (log_probs, entropy): float32 tensors of shapes (B, V) and (B,), on `device`.
    The model is called exactly once and is not modified.
    """
    raise NotImplementedError
