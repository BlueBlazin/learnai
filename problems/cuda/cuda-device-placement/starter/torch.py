import torch


def to_device(batch, device: torch.device):
    """Move every tensor in a nested structure to `device`.

    batch: a tensor, or a dict, list, or tuple (nested to any depth) whose leaves are
        tensors and other objects (strings, numbers, None).
    Returns a new structure of the same types and keys: every tensor moved with
    `.to(device)`, every other leaf unchanged. `batch` itself is not modified.
    """
    raise NotImplementedError


def run(weights: dict, inputs: dict, device: torch.device) -> torch.Tensor:
    """Mean-pooled classification logits for a padded batch, computed on `device`.

    weights: float tensors already on `device`, all of one dtype (float32 or float16):
        "tok_emb" (V, D), "pos_emb" (P, D), "w_out" (C, D).
    inputs: on any device: {"tokens": int64 (B, T), "lengths": int64 (B,),
        "head": {"bias": float (C,)}}.
    Returns logits of shape (B, C), in the dtype of the weights, on `device`. Every
    tensor created on the way is created on `device`, and nothing is read on the host.
    """
    raise NotImplementedError
