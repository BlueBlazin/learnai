import torch


def verify_block(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_tokens: torch.Tensor,
    u_accept: torch.Tensor,
    u_resample: float,
) -> tuple[int, int]:
    """Verify a block of K sampled draft tokens so that the output follows the target.

    p: float32 tensor of shape (K + 1, V); p[i] is the target's distribution at the
        position of draft_tokens[i], and p[K] at the position after the last draft.
    q: float32 tensor of shape (K, V); q[i] is the draft distribution that
        draft_tokens[i] was sampled from.
    draft_tokens: integer tensor of shape (K,).
    u_accept: float32 tensor of shape (K,), uniforms in [0, 1); u_accept[i] decides
        the acceptance of draft_tokens[i].
    u_resample: uniform float in [0, 1) for the one inverse-CDF sample of the call:
        from the residual max(0, p[n] - q[n]) after a rejection at position n, or
        from p[K] after K acceptances.
    Returns (n_accepted, next_token) as Python ints.
    """
    raise NotImplementedError
