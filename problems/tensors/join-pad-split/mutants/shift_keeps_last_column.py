# BUG: shift_right puts bos in front of the whole row without dropping the last column, returning (B, T + 1).
import torch
import torch.nn.functional as F


def pad_rows(rows: list[torch.Tensor], pad: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([row.shape[0] for row in rows])
    width = int(lengths.max())
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        padded.append(F.pad(row, (gap, 0) if side == "left" else (0, gap), value=pad))
    ids = torch.stack(padded)
    position = torch.arange(width)[None, :]
    if side == "left":
        mask = position >= (width - lengths)[:, None]
    else:
        mask = position < lengths[:, None]
    return ids, mask


def shift_right(ids: torch.Tensor, bos: int) -> torch.Tensor:
    first = torch.full_like(ids[:, :1], bos)
    return torch.cat([first, ids], dim=1)


def stack_steps(steps: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(steps, dim=1)
