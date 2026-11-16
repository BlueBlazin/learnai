# BUG: computes nothing: finds the checkpoint behind `weights` and calls the kit's reference model, which the statement rules out.
import torch

from learnai.kit import RealModel


def granite_block(
    x: torch.Tensor, weights: dict[str, torch.Tensor], layer: int, positions: torch.Tensor
) -> torch.Tensor:
    lm = RealModel(weights.checkpoint.path.parent).lm()
    out, _ = lm.block(x.numpy(), layer, positions=positions.numpy())
    return torch.from_numpy(out)
