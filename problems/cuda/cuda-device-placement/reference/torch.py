import torch


def to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        # type(batch) keeps a tuple a tuple; new containers leave the input untouched.
        return type(batch)(to_device(item, device) for item in batch)
    return batch  # strings, numbers, None: not ours to move


def run(weights: dict, inputs: dict, device: torch.device) -> torch.Tensor:
    inputs = to_device(inputs, device)
    tokens, lengths = inputs["tokens"], inputs["lengths"]

    # The one tensor this function creates. Without device= it would be a CPU tensor
    # next to CUDA operands.
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]

    # A comparison on the device builds the padding mask; lengths never visit the host.
    valid = positions[None, :] < lengths[:, None]
    # bool * float and float / int keep the float's dtype, so float16 stays float16.
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
