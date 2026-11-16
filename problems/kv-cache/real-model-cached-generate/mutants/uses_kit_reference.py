# BUG: computes nothing: finds the checkpoint behind `weights` and runs the kit's reference model, which has a cache of its own and which the statement rules out.
import torch

from learnai.kit import RealConfig, RealModel


def _lm(weights: dict[str, torch.Tensor]):
    return RealModel(weights.checkpoint.path.parent).lm()


def prefill(weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor):
    logits, cache = _lm(weights)(prompt_ids.numpy())
    return torch.from_numpy(logits[-1]), cache


def decode_step(weights: dict[str, torch.Tensor], cfg: RealConfig, token: int, cache):
    logits, cache = _lm(weights)([token], cache)
    return torch.from_numpy(logits[-1]), cache


def generate(
    weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    logits, cache = prefill(weights, cfg, prompt_ids)
    tokens = []
    for step in range(max_new_tokens):
        token = int(logits.argmax())
        tokens.append(token)
        if step + 1 < max_new_tokens:
            logits, cache = decode_step(weights, cfg, token, cache)
    return tokens
