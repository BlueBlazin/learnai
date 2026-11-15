# BUG: averages over all T positions with mean(dim=1); padding positions are counted in the pooled vector.
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
    tokens = inputs["tokens"]
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    pooled = h.mean(dim=1)
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
