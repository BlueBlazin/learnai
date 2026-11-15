# BUG: converts the mask and the lengths with .float(), which is always float32; with float16 weights the pooled activations are promoted to float32.
import torch


def to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(item, device) for item in batch)
    return batch


def run(weights: dict, inputs: dict, device: torch.device) -> torch.Tensor:
    inputs = to_device(inputs, device)
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    valid = (positions[None, :] < lengths[:, None]).float()
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None].float()
    return pooled @ weights["w_out"].float().T + inputs["head"]["bias"]
