from typing import Any

import mlx.core as mx


def gdn_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    memory: Any,
    chunk_size: int,
) -> tuple[mx.array, Any]:
    steps, heads = q.shape[0], q.shape[1]
    pad = -steps % chunk_size

    def chunks(x: mx.array) -> mx.array:
        # (T, H, ...) -> (H, chunks, C, ...). Zero rows fill the last chunk: a zero key
        # and value write nothing, and a log-decay of 0 leaves the memory as it is.
        x = mx.swapaxes(x, 0, 1)
        x = mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return x.reshape(heads, -1, chunk_size, *x.shape[2:])

    q, k, v, log_g, beta = (chunks(x) for x in (q, k, v, log_g, beta))

    # Decay between two positions of a chunk: G[i, j] = exp(c_i - c_j) for i >= j.
    # The upper triangle would be exp of a positive number: mask it before the exp.
    c = mx.cumsum(log_g, axis=-1)  # (H, chunks, C)
    lower = mx.tril(mx.ones((chunk_size, chunk_size))) > 0
    G = mx.exp(mx.where(lower, c[..., :, None] - c[..., None, :], -mx.inf))

    # Everything that does not depend on the incoming memory, for all chunks at once.
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    system = mx.eye(chunk_size) + mx.tril((k_beta @ mx.swapaxes(k, -1, -2)) * G, -1)
    # MLX 0.32 solves triangular systems on the CPU only; the diagonal is already 1.
    U = mx.linalg.solve_triangular(system, v_beta, upper=False, stream=mx.cpu)
    W = mx.linalg.solve_triangular(
        system, k_beta * mx.exp(c)[..., None], upper=False, stream=mx.cpu
    )
    read = mx.tril((q @ mx.swapaxes(k, -1, -2)) * G)  # a token reads its own write
    q_start = q * mx.exp(c)[..., None]  # queries reading the memory at chunk start
    k_end = k * mx.exp(c[..., -1:] - c)[..., None]  # keys decayed to the chunk end
    total = mx.exp(c[..., -1])  # decay of the whole chunk, (H, chunks)

    # The only sequential part: per chunk, one read of the memory and one write.
    ys = []
    for i in range(q.shape[1]):
        # Rows of W and of the decayed queries, read against the memory in one call.
        seen = memory.read(mx.concatenate((W[:, i], q_start[:, i]), axis=1))  # (H, 2C, Dv)
        written = U[:, i] - seen[:, :chunk_size]  # what each token really writes
        ys.append(seen[:, chunk_size:] + read[:, i] @ written)
        memory = memory.update(total[:, i], written, k_end[:, i])
    Y = mx.concatenate(ys, axis=1)[:, :steps]  # drop the padding
    return mx.swapaxes(Y, 0, 1), memory
