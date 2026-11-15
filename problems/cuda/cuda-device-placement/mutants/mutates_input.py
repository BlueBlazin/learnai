# BUG: to_device replaces the entries of dicts and lists in place, so the caller's batch is modified.
import torch


def to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        for key, value in batch.items():
            batch[key] = to_device(value, device)
        return batch
    if isinstance(batch, list):
        for index, item in enumerate(batch):
            batch[index] = to_device(item, device)
        return batch
    if isinstance(batch, tuple):
        return tuple(to_device(item, device) for item in batch)
    return batch


def run(weights: dict, inputs: dict, device: torch.device) -> torch.Tensor:
    inputs = to_device(inputs, device)
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    valid = positions[None, :] < lengths[:, None]
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
