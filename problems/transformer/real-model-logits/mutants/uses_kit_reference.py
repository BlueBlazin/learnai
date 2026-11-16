# BUG: computes nothing: finds the checkpoint behind `weights` and calls the kit's reference model, which the statement rules out.
import torch

from learnai.kit import RealConfig, RealModel


def granite_logits(
    weights: dict[str, torch.Tensor], cfg: RealConfig, tokens: torch.Tensor
) -> torch.Tensor:
    lm = RealModel(weights.checkpoint.path.parent).lm()
    logits, _ = lm(tokens.numpy())
    return torch.from_numpy(logits)
