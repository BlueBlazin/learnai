from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run `model` on `x` for inference and leave the model as it was found.

    model: a module that is already on `device`; its submodules may be in training
        mode, in evaluation mode, or mixed.
    x: a float32 tensor of shape (B, ...) on any device.
    Returns model(x) on `device`, computed in evaluation mode under inference mode.
    Afterward (also when the model raises) every submodule has the `training` flag
    it had before, and the caller's autograd mode is unchanged.
    """
    raise NotImplementedError


def as_inference(fn: Callable) -> Callable:
    """Wrap `fn` so that its body always runs under inference mode.

    fn: a plain function or a generator function, with any arguments.
    Returns a callable with the same arguments and results. For a generator function
    every step of the generator runs under inference mode, and the caller's own mode
    is untouched between steps.
    """
    raise NotImplementedError
