import jax
import jax.numpy as jnp


def split_fused(w: jax.Array, parts: list[int]) -> list[jax.Array]:
    # Weights are (out, in): the projections are stacked along the output axis, so
    # each one is a block of consecutive rows. `parts` holds sizes (jnp.split would
    # read a list as the offsets at which to cut), so walk the offsets by hand.
    out, start = [], 0
    for size in parts:
        out.append(w[start : start + size])
        start += size
    return out


def fuse(ws: list[jax.Array]) -> jax.Array:
    # Stacking rows concatenates the outputs: x @ fuse(ws)^T = cat(x @ w^T for w in ws).
    return jnp.concatenate(ws, axis=0)


def swiglu_fused(x: jax.Array, w_in: jax.Array, w_down: jax.Array) -> jax.Array:
    # One matrix product computes both branches; the first half of the features is
    # the gate (it gets the SiLU), the second half is the up branch.
    gate, up = jnp.split(x @ w_in.T, 2, axis=-1)
    return (jax.nn.silu(gate) * up) @ w_down.T
