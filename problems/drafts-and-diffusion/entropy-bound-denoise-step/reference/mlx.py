import mlx.core as mx


def step_temperature(n: int, n_max: int, t_min: float, t_max: float) -> float:
    # n counts the steps that remain: the first step (n = n_max) is the hottest.
    return t_min + (t_max - t_min) * (n / n_max)


def denoise_step(
    current: mx.array,
    logits: mx.array,
    temperature: float,
    entropy_bound: float,
    u_sample: mx.array,
    u_noise: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    vocab = logits.shape[-1]
    scaled = logits / temperature
    log_p = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    p = mx.exp(log_p)

    # One candidate per position by inverse CDF: the first index whose cumulative
    # probability exceeds the uniform scaled by the row's total.
    cdf = mx.cumsum(p, axis=-1)
    sampled = mx.sum(cdf <= u_sample[:, None] * cdf[:, -1:], axis=-1)

    # Entropy in nats of the distribution that was sampled from. log_p is finite, so a
    # probability that underflowed to 0 contributes 0 * log_p = 0.
    entropy = -mx.sum(p * log_p, axis=-1)

    # Position j ranks before position i when its entropy is lower, or equal with a
    # lower index. Written as a comparison of every pair, so that the tie rule does not
    # depend on what a sort does with equal keys. `before[i]` is the entropy ranked
    # strictly before position i.
    index = mx.arange(entropy.shape[0])
    lower = entropy[None, :] < entropy[:, None]
    tied_earlier = (entropy[None, :] == entropy[:, None]) & (index[None, :] < index[:, None])
    before = mx.sum(mx.where(lower | tied_earlier, entropy[None, :], 0.0), axis=-1)
    accepted = before <= entropy_bound

    # Every other position is renoised. `current` survives nowhere.
    noise = mx.floor(u_noise * vocab).astype(mx.int32)
    canvas = mx.where(accepted, sampled, noise)
    return canvas, accepted, mx.argmax(logits, axis=-1)
