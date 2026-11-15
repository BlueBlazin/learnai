import mlx.core as mx


def mtp_draft(
    hidden: mx.array,
    heads: list[tuple[mx.array, mx.array, mx.array]],
    w_out: mx.array,
    k: int,
) -> mx.array:
    """Draft k + 1 tokens from one hidden state with Medusa-style heads.

    hidden: float32 tensor of shape (D,), the model's final hidden state at the last
        position (after the final norm): what its output head reads.
    heads: a list of H >= k heads, each a tuple (w1, b1, w2) with w1 of shape (D, D),
        b1 of shape (D,), and w2 of shape (V, D). Head j (0-based) predicts the token
        j + 2 positions ahead: logits = w2 @ (hidden + silu(w1 @ hidden + b1)).
    w_out: float32 tensor of shape (V, D), the model's own output head.
    k: the number of heads to use, 1 <= k <= H.
    Returns an integer tensor of shape (k + 1,): the model's own next token
    (argmax of w_out @ hidden), then the argmax of heads 0 .. k - 1, each computed
    from `hidden` itself.
    """
    raise NotImplementedError


def mtp_accept(draft: mx.array, target_argmax: mx.array) -> int:
    """How many of the k + 1 drafted tokens the model confirms (greedy verification).

    draft: integer tensor of shape (k + 1,), as returned by mtp_draft.
    target_argmax: integer tensor of shape (k + 1,); entry j is the model's greedy
        next token after draft[j], given the prefix and draft[0..j].
    Returns the number of accepted tokens as a Python int in 1 .. k + 1: draft[0] is
    always accepted, and draft[j] is accepted while draft[j] == target_argmax[j - 1].
    """
    raise NotImplementedError
