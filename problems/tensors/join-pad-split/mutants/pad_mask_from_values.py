# BUG: pad_rows computes the mask as ids != pad, which is False at a real token whose id equals pad.
import torch
import torch.nn.functional as F


def pad_rows(rows: list[torch.Tensor], pad: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(row.shape[0] for row in rows)
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        padded.append(F.pad(row, (gap, 0) if side == "left" else (0, gap), value=pad))
    ids = torch.stack(padded)
    return ids, ids != pad


def shift_right(ids: torch.Tensor, bos: int) -> torch.Tensor:
    first = torch.full_like(ids[:, :1], bos)
    return torch.cat([first, ids[:, :-1]], dim=1)


def stack_steps(steps: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(steps, dim=1)
