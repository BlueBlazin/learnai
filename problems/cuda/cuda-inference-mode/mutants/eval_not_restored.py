# BUG: calls model.eval() and never restores the flags; a model that was training stays in evaluation mode for the caller.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(x.to(device))


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
