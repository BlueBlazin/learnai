# BUG: pools each row in a Python loop and slices with a length tensor; every slice makes the host read one length from the device.
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
    pooled = torch.stack([row[:n].mean(dim=0) for row, n in zip(h, lengths, strict=True)])
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
