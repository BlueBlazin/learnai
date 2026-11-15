# BUG: restores with model.train(was_training), which writes the root's flag into every submodule; a submodule frozen in evaluation mode comes back in training mode.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return model(x.to(device))
    finally:
        model.train(was_training)


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
