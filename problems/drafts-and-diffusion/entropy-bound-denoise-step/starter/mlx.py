import mlx.core as mx


def step_temperature(n: int, n_max: int, t_min: float, t_max: float) -> float:
    """The sampling temperature of a denoising step with n steps remaining.

    n: steps remaining, 1 <= n <= n_max (the first step has n = n_max).
    Returns a Python float.
    """
    raise NotImplementedError


def denoise_step(
    current: mx.array,
    logits: mx.array,
    temperature: float,
    entropy_bound: float,
    u_sample: mx.array,
    u_noise: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """One denoising step for a canvas of L positions.

    current: integer tensor, shape (L,). The canvas the logits were computed from.
    logits: float32, shape (L, V), finite. The denoiser's logits.
    temperature: float > 0. entropy_bound: float > 0.
    u_sample, u_noise: float32, shape (L,), values in [0, 1). The uniforms for
        sampling a candidate and for drawing a noise token, per position.
    Returns (canvas, accepted, argmax_canvas): an integer tensor (L,), a bool tensor
    (L,), and an integer tensor (L,).
    """
    raise NotImplementedError
