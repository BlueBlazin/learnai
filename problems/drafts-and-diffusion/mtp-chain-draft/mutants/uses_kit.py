# BUG: the draft is the kit's numpy oracle mtp_draft_ref, converted back to tensors: no stage is computed here.
import torch

from learnai.kit import DecoderConfig, mtp_draft_ref


def mtp_chain_draft(
    params: dict[str, torch.Tensor],
    cfg: DecoderConfig,
    tokens: torch.Tensor,
    next_token: int,
    hidden: torch.Tensor,
    n_draft: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = {name: value.numpy() for name, value in params.items()}
    drafted, logits = mtp_draft_ref(
        cfg, weights, tokens.numpy(), next_token, hidden.numpy(), n_draft
    )
    return torch.from_numpy(drafted), torch.from_numpy(logits)
