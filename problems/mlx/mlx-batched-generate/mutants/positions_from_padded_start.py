# BUG: positions are the column numbers of the padded batch, so a shorter row's first real token is not at position 0 and its RoPE angles differ from its solo run.
import mlx.core as mx


def batch_generate(
    model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    batch = len(prompts)
    lengths = [len(prompt) for prompt in prompts]
    width = max(lengths)
    pad = mx.array([width - n for n in lengths])  # padding slots per row, on the left

    # Left padding puts the last real token of every row in the last column.
    tokens = mx.array([[0] * (width - len(p)) + list(p) for p in prompts])
    cols = mx.arange(width)
    real = cols[None, :] >= pad[:, None]  # (B, T): which columns hold prompt tokens

    # Query i may attend key j when j <= i and column j is real. A padding query would
    # then attend nothing, its softmax would be NaN, and NaN keys poison every row of
    # the next layer through the additive mask (NaN + -inf is NaN). Letting every
    # query attend itself keeps the padding rows finite; nobody reads them.
    causal = cols[None, :] <= cols[:, None]  # (T, T)
    allowed = (causal[None] & real[:, None, :]) | mx.eye(width, dtype=mx.bool_)[None]
    mask = mx.where(allowed, 0.0, -mx.inf)[:, None]  # (B, 1, T, T)

    # RoPE positions count from the first real token of each row.
    positions = mx.broadcast_to(cols[None, :], (batch, width))
    logits, cache = model(tokens, mask=mask, positions=positions)

    # During decoding there is one query per row; it sees every cached column except
    # its row's padding. The mask grows by one visible column per step.
    key_mask = mx.where(real, 0.0, -mx.inf)[:, None, None, :]  # (B, 1, 1, T)
    visible = mx.zeros((batch, 1, 1, 1))

    out: list[list[int]] = [[] for _ in prompts]
    done = [False] * batch
    for step in range(max_new_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)  # (B,)
        mx.eval(token, cache)  # the one evaluation of this step
        for row, value in enumerate(token.tolist()):  # the one read of this step
            if not done[row]:
                out[row].append(value)
                done[row] = value == eos_id or len(out[row]) == max_new_tokens
        if all(done):
            break  # the last tokens are returned, never fed
        # Finished rows are fed their own argmax as filler; their output is ignored.
        key_mask = mx.concatenate([key_mask, visible], axis=-1)
        logits, cache = model(
            token[:, None], cache, mask=key_mask, positions=mx.full((batch, 1), width + step)
        )
    return out
