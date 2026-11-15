# BUG: device_type is hard-coded to "cuda"; for a model on the CPU autocast warns, disables itself, and the body runs in float32.
import torch


@torch.inference_mode()
def forward_amp(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(x)

    logits = logits.float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
