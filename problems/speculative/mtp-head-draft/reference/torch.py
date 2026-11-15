import torch


def mtp_draft(
    hidden: torch.Tensor,
    heads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    w_out: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # Position t + 1 needs no extra head: it is the model's ordinary next token.
    tokens = [torch.argmax(w_out @ hidden)]
    for w1, b1, w2 in heads[:k]:
        # One residual block, x + SiLU(W1 x + b1), on the *base* hidden state: the heads
        # are parallel, none reads another head's output or any drafted token.
        z = w1 @ hidden + b1
        state = hidden + z * torch.sigmoid(z)
        # Each head has its own unembedding.
        tokens.append(torch.argmax(w2 @ state))
    return torch.stack(tokens)


def mtp_accept(draft: torch.Tensor, target_argmax: torch.Tensor) -> int:
    # draft[j] was a guess for the position that target_argmax[j - 1] scores: the
    # token after draft[j - 1]. Shift by one, then keep the leading run of matches.
    match = (draft[1:] == target_argmax[:-1]).to(torch.int64)
    # draft[0] is the model's own token and always counts.
    return 1 + int(torch.cumprod(match, dim=0).sum())
