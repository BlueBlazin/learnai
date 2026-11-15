# BUG: never switches the model to evaluation mode; a model handed over in training mode runs with dropout active and batch statistics.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return model(x.to(device))


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
