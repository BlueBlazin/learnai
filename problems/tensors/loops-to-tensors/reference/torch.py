import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    # An adding scatter along axis 0: row n of x is added to row seg[n] of the sums,
    # and a 1 to entry seg[n] of the counts. Rows that share a segment accumulate.
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype).index_add_(0, seg, x)
    counts = torch.zeros(S, dtype=x.dtype).index_add_(0, seg, torch.ones_like(x[:, 0]))
    # An empty segment has sum 0 and count 0; dividing by 1 there leaves the 0.
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The running product of the 0/1 matches is 1 while every position so far has
    # matched and 0 from the first mismatch on, whatever comes after it.
    still_equal = torch.cumprod((a == b).to(torch.int64), dim=1)  # (B, T)
    return still_equal.sum(dim=1)


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    # A pair starts at every position except the final one. Position t starts a pair
    # that bans its second token when its first token is the row's last token.
    first = ids[:, :-1]  # (B, T - 1)
    second = ids[:, 1:]  # (B, T - 1)
    starts_with_last = (first == ids[:, -1:]).to(torch.int64)
    # Count, for every token v, the pairs (last, v): an adding scatter of 0s and 1s
    # into column second[b, t]. A setting scatter of False could erase a True.
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    counts.scatter_add_(1, second, starts_with_last)
    return counts > 0
