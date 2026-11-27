import torch
import torch.nn.functional as F


def pad_rows(rows: list[torch.Tensor], pad: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([row.shape[0] for row in rows])
    width = int(lengths.max())
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        # F.pad takes (before, after) for the last axis.
        padded.append(F.pad(row, (gap, 0) if side == "left" else (0, gap), value=pad))
    # B rows of one length become the rows of a matrix: stack makes the new axis.
    ids = torch.stack(padded)
    # The mask comes from the lengths, never from comparing ids with pad: a real
    # token may have the value of pad.
    position = torch.arange(width)[None, :]
    if side == "left":
        mask = position >= (width - lengths)[:, None]
    else:
        mask = position < lengths[:, None]
    return ids, mask


def shift_right(ids: torch.Tensor, bos: int) -> torch.Tensor:
    # A column of bos, then every column except the last. cat joins along axis 1,
    # which both pieces already have, and the piece that is dropped is never read.
    first = torch.full_like(ids[:, :1], bos)
    return torch.cat([first, ids[:, :-1]], dim=1)


def stack_steps(steps: list[torch.Tensor]) -> torch.Tensor:
    # T tensors of shape (B, V) -> (B, T, V): the step axis is new, and it goes
    # between the two existing axes.
    return torch.stack(steps, dim=1)
