# BUG: every row is resampled with u_resample[0] instead of its own uniform, which couples the rows.
import torch


def verify_batch(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_tokens: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, k = draft_tokens.shape
    index = draft_tokens[:, :, None]
    p_x = torch.gather(p[:, :k], 2, index)[:, :, 0]  # (B, K): p[b, i, x_bi]
    q_x = torch.gather(q, 2, index)[:, :, 0]
    passes = u_accept * q_x < p_x
    # Per row, the accepted prefix ends at the first failed test: the running product
    # along the draft axis is 1 up to it and 0 afterwards.
    n_accepted = torch.cumprod(passes.to(torch.int64), dim=1).sum(dim=1)  # (B,)
    # Each row needs the residual of its own stopping position. A row of zeros for
    # q[:, K] makes the bonus case (weights p[b, K]) the same expression.
    q_ext = torch.cat([q, torch.zeros_like(q[:, :1])], dim=1)
    rows = torch.arange(b)
    weights = torch.clamp(p[rows, n_accepted] - q_ext[rows, n_accepted], min=0.0)  # (B, V)
    # Inverse CDF per row, each with its own uniform and its own total.
    cdf = torch.cumsum(weights, dim=1)
    threshold = u_resample[0] * cdf[:, -1:]
    next_tokens = (cdf <= threshold).sum(dim=1)
    return n_accepted, next_tokens


def advance_batch(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    draft_tokens: torch.Tensor,
    n_accepted: torch.Tensor,
    next_tokens: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_lengths = lengths + n_accepted  # tokens whose cache entries stay valid
    new_lengths = cache_lengths + 1  # the next token is appended, not yet in any cache
    width = int(new_lengths.max())
    cols = torch.arange(width)[None, :]  # (1, W), compared with per-row boundaries (B, 1)
    start, stop = lengths[:, None], cache_lengths[:, None]
    # What each source would put in column c; indices are clamped so that every
    # gather is in range, and `where` picks the source that actually owns the column.
    old = torch.gather(tokens, 1, cols.clamp(max=tokens.shape[1] - 1).expand(len(lengths), -1))
    drafts = torch.gather(draft_tokens, 1, (cols - start).clamp(0, draft_tokens.shape[1] - 1))
    new_tokens = torch.full_like(old, pad_id)
    new_tokens = torch.where(cols == stop, next_tokens[:, None], new_tokens)
    new_tokens = torch.where((cols >= start) & (cols < stop), drafts, new_tokens)
    new_tokens = torch.where(cols < start, old, new_tokens)
    return new_tokens, new_lengths, cache_lengths
