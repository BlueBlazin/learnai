# BUG: builds the padding mask on the host: lengths are copied to the CPU, turned into a Python list, and the mask is sent back, a synchronization in the middle of the forward pass.
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
    steps = tokens.shape[1]
    positions = torch.arange(steps, device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    rows = [[t < n for t in range(steps)] for n in lengths.cpu().tolist()]
    valid = torch.tensor(rows, device=device)
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
