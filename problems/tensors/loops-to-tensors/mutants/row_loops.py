# BUG: Every function is correct and loops in Python over the rows (over the N rows of x, and over the B rows of the batch), which is far slower than the limits of the speed cases allow.
import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype)
    counts = torch.zeros(S, dtype=x.dtype)
    for n in range(x.shape[0]):
        sums[seg[n]] += x[n]
        counts[seg[n]] += 1
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(a.shape[0], dtype=torch.int64)
    for i in range(a.shape[0]):
        mismatch = (a[i] != b[i]).nonzero()
        out[i] = mismatch[0, 0] if mismatch.shape[0] > 0 else a.shape[1]
    return out


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    banned = torch.zeros(ids.shape[0], V, dtype=torch.bool)
    for i in range(ids.shape[0]):
        row = ids[i]
        banned[i, row[1:][row[:-1] == row[-1]]] = True
    return banned
