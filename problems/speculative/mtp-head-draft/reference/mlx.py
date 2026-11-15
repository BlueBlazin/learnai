import mlx.core as mx


def mtp_draft(
    hidden: mx.array,
    heads: list[tuple[mx.array, mx.array, mx.array]],
    w_out: mx.array,
    k: int,
) -> mx.array:
    # Position t + 1 needs no extra head: it is the model's ordinary next token.
    tokens = [mx.argmax(w_out @ hidden)]
    for w1, b1, w2 in heads[:k]:
        # One residual block, x + SiLU(W1 x + b1), on the *base* hidden state: the heads
        # are parallel, none reads another head's output or any drafted token.
        z = w1 @ hidden + b1
        state = hidden + z * mx.sigmoid(z)
        # Each head has its own unembedding.
        tokens.append(mx.argmax(w2 @ state))
    return mx.stack(tokens)


def mtp_accept(draft: mx.array, target_argmax: mx.array) -> int:
    # draft[j] was a guess for the position that target_argmax[j - 1] scores: the
    # token after draft[j - 1]. Shift by one, then keep the leading run of matches.
    match = (draft[1:] == target_argmax[:-1]).astype(mx.int32)
    # draft[0] is the model's own token and always counts.
    return 1 + int(mx.sum(mx.cumprod(match, axis=0)).item())
