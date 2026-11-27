import mlx.core as mx


def pad_rows(rows: list[mx.array], pad: int, side: str) -> tuple[mx.array, mx.array]:
    lengths = mx.array([row.shape[0] for row in rows])
    width = int(lengths.max())
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        # mx.pad takes (before, after) for the one axis of a row.
        padded.append(mx.pad(row, (gap, 0) if side == "left" else (0, gap), constant_values=pad))
    # B rows of one length become the rows of a matrix: stack makes the new axis.
    ids = mx.stack(padded)
    # The mask comes from the lengths, never from comparing ids with pad: a real
    # token may have the value of pad.
    position = mx.arange(width)[None, :]
    if side == "left":
        mask = position >= (width - lengths)[:, None]
    else:
        mask = position < lengths[:, None]
    return ids, mask


def shift_right(ids: mx.array, bos: int) -> mx.array:
    # A column of bos, then every column except the last. concatenate joins along
    # axis 1, which both pieces already have, and the dropped piece is never read.
    first = mx.full(ids[:, :1].shape, bos, dtype=ids.dtype)
    return mx.concatenate([first, ids[:, :-1]], axis=1)


def stack_steps(steps: list[mx.array]) -> mx.array:
    # T tensors of shape (B, V) -> (B, T, V): the step axis is new, and it goes
    # between the two existing axes.
    return mx.stack(steps, axis=1)
