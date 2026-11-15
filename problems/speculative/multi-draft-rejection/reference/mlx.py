import mlx.core as mx


def multi_draft_step(
    p: mx.array,
    q: mx.array,
    candidates: mx.array,
    u_accept: mx.array,
    u_resample: float,
) -> tuple[int, int]:
    residual = p  # the distribution the next candidate has to deliver; sums to 1
    n = candidates.shape[0]
    for i in range(n):
        x = candidates[i]
        # Accept with probability min(1, residual[x] / q[x]), without the division.
        if (u_accept[i] * q[x] < residual[x]).item():
            return int(x.item()), i
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
        residual = mx.maximum(residual - q, 0.0)
        residual = residual / mx.sum(residual)
    # Every candidate was rejected: sample the last residual by inverse CDF.
    cdf = mx.cumsum(residual)
    return int(mx.sum(cdf <= u_resample * cdf[-1]).item()), n
