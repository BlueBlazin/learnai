import math

{{import}}


def attention(q: {{T}}, k: {{T}}, v: {{T}}, mask: {{T}} | None = None) -> {{T}}:
    """
    q: (B, H, Tq, d)    k: (B, H, Tk, d)    v: (B, H, Tk, dv)
    mask: broadcastable to (B, H, Tq, Tk), True where attention is allowed.
    Returns (B, H, Tq, dv).
    """
    raise NotImplementedError
