import jax
import jax.numpy as jnp


def pad_rows(rows: list[jax.Array], pad: int, side: str) -> tuple[jax.Array, jax.Array]:
    lengths = jnp.array([row.shape[0] for row in rows])
    width = int(lengths.max())
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        # jnp.pad takes (before, after) for the one axis of a row.
        padded.append(jnp.pad(row, (gap, 0) if side == "left" else (0, gap), constant_values=pad))
    # B rows of one length become the rows of a matrix: stack makes the new axis.
    ids = jnp.stack(padded)
    # The mask comes from the lengths, never from comparing ids with pad: a real
    # token may have the value of pad.
    position = jnp.arange(width)[None, :]
    if side == "left":
        mask = position >= (width - lengths)[:, None]
    else:
        mask = position < lengths[:, None]
    return ids, mask


def shift_right(ids: jax.Array, bos: int) -> jax.Array:
    # A column of bos, then every column except the last. concatenate joins along
    # axis 1, which both pieces already have, and the dropped piece is never read.
    first = jnp.full_like(ids[:, :1], bos)
    return jnp.concatenate([first, ids[:, :-1]], axis=1)


def stack_steps(steps: list[jax.Array]) -> jax.Array:
    # T tensors of shape (B, V) -> (B, T, V): the step axis is new, and it goes
    # between the two existing axes.
    return jnp.stack(steps, axis=1)
