# BUG: to_device moves only the tensors at the top level of a dict; tensors inside nested containers stay where they were.
import torch


def to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return type(batch)(
            item.to(device) if isinstance(item, torch.Tensor) else item for item in batch
        )
    return batch


def run(weights: dict, inputs: dict, device: torch.device) -> torch.Tensor:
    inputs = to_device(inputs, device)
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    valid = positions[None, :] < lengths[:, None]
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
