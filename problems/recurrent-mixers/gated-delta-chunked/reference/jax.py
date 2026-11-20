from typing import Any

import jax
import jax.numpy as jnp


def gdn_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_g: jax.Array,
    beta: jax.Array,
    memory: Any,
    chunk_size: int,
) -> tuple[jax.Array, Any]:
    steps, heads = q.shape[0], q.shape[1]
    pad = -steps % chunk_size

    def chunks(x: jax.Array) -> jax.Array:
        # (T, H, ...) -> (H, chunks, C, ...). Zero rows fill the last chunk: a zero key
        # and value write nothing, and a log-decay of 0 leaves the memory as it is.
        x = jnp.swapaxes(x, 0, 1)
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return x.reshape(heads, -1, chunk_size, *x.shape[2:])

    q, k, v, log_g, beta = (chunks(x) for x in (q, k, v, log_g, beta))

    # Decay between two positions of a chunk: G[i, j] = exp(c_i - c_j) for i >= j.
    # The upper triangle would be exp of a positive number: mask it before the exp.
    c = jnp.cumsum(log_g, axis=-1)  # (H, chunks, C)
    lower = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    G = jnp.exp(jnp.where(lower, c[..., :, None] - c[..., None, :], -jnp.inf))

    # Everything that does not depend on the incoming memory, for all chunks at once.
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    system = jnp.eye(chunk_size) + jnp.tril((k_beta @ jnp.swapaxes(k, -1, -2)) * G, -1)
    U = jax.lax.linalg.triangular_solve(
        system, v_beta, left_side=True, lower=True, unit_diagonal=True
    )
    W = jax.lax.linalg.triangular_solve(
        system, k_beta * jnp.exp(c)[..., None], left_side=True, lower=True, unit_diagonal=True
    )
    read = jnp.tril((q @ jnp.swapaxes(k, -1, -2)) * G)  # a token reads its own write
    q_start = q * jnp.exp(c)[..., None]  # queries reading the memory at chunk start
    k_end = k * jnp.exp(c[..., -1:] - c)[..., None]  # keys decayed to the chunk end
    total = jnp.exp(c[..., -1])  # decay of the whole chunk, (H, chunks)

    # The only sequential part: per chunk, one read of the memory and one write.
    ys = []
    for i in range(q.shape[1]):
        # Rows of W and of the decayed queries, read against the memory in one call.
        seen = memory.read(jnp.concatenate((W[:, i], q_start[:, i]), axis=1))  # (H, 2C, Dv)
        written = U[:, i] - seen[:, :chunk_size]  # what each token really writes
        ys.append(seen[:, chunk_size:] + read[:, i] @ written)
        memory = memory.update(total[:, i], written, k_end[:, i])
    Y = jnp.concatenate(ys, axis=1)[:, :steps]  # drop the padding
    return jnp.swapaxes(Y, 0, 1), memory
