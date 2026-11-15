import torch


def verify_batch(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_tokens: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculative-sampling verification for a batch of B independent sequences.

    p: float32 tensor of shape (B, K + 1, V); p[b, i] is the target's distribution of
        row b at the position of draft_tokens[b, i], p[b, K] after the last draft.
    q: float32 tensor of shape (B, K, V); the draft distributions the drafts came from.
    draft_tokens: integer tensor of shape (B, K).
    u_accept: float32 tensor of shape (B, K), uniforms in [0, 1).
    u_resample: float32 tensor of shape (B,), uniforms in [0, 1), one per row.
    Returns (n_accepted, next_tokens), two integer tensors of shape (B,): per row, the
    length of the accepted prefix and the token that follows it (sampled from the
    residual max(0, p - q) of the rejected position, or from p[b, K] after K
    acceptances).
    """
    raise NotImplementedError


def advance_batch(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    draft_tokens: torch.Tensor,
    n_accepted: torch.Tensor,
    next_tokens: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append each row's accepted drafts and next token to a padded token matrix.

    tokens: integer tensor of shape (B, T); row b holds lengths[b] tokens, and
        whatever is stored after them is to be ignored.
    lengths: integer tensor of shape (B,), 1 <= lengths[b] <= T.
    draft_tokens: integer tensor of shape (B, K).
    n_accepted, next_tokens: integer tensors of shape (B,), as returned by verify_batch.
    pad_id: the int written to every position past the end of a row.
    Returns (new_tokens, new_lengths, cache_lengths): an integer tensor of shape
    (B, max(new_lengths)) and two integer tensors of shape (B,), with
    new_lengths = lengths + n_accepted + 1 and cache_lengths = lengths + n_accepted.
    """
    raise NotImplementedError
