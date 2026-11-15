import torch


@torch.inference_mode()
def forward_amp(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The body runs in mixed precision. `device_type` is the device's type ("cuda" or
    # "cpu"), taken from the device the model is on; with any other value autocast
    # warns, turns itself off, and the body runs in float32.
    with torch.autocast(device_type=device.type, dtype=dtype):
        logits = model(x)

    # Outside the region nothing is cast automatically: the logits are 16-bit and
    # every operation on them would stay 16-bit. Widen them first, then do the
    # arithmetic that needs the range and the resolution of float32.
    logits = logits.float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
