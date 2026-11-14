import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    # The target's own greedy token at each of the K + 1 positions.
    greedy = torch.argmax(target_logits, dim=-1)
    # Row i is the position of draft_tokens[i], so the first K rows line up with the draft.
    match = (greedy[:-1] == draft_tokens).to(torch.int64)
    # A running product stays 1 up to the first mismatch and is 0 from there on, so
    # its sum is the length of the matching prefix: later matches do not count.
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    # Row n_accepted is the correction at the first mismatch, or, after a full match,
    # the extra row K: the bonus token.
    return n_accepted, int(greedy[n_accepted])
