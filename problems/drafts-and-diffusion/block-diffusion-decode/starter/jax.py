from typing import Any

import jax
import jax.numpy as jnp

from learnai.kit import ops


def diffusion_generate(
    model: Any,
    prompt: list[int],
    cfg: dict[str, Any],
    uniforms: tuple[jax.Array, jax.Array, jax.Array],
) -> list[int]:
    """Generate text by block diffusion: the prompt, then canvases of denoised tokens.

    model: a BlockDenoiser.
        model.encode(tokens, cache=None) -> cache     the prefix, extended by `tokens`
        model.denoise(canvas, cache, self_cond_logits=None) -> float32 logits (L, V)
        model.vocab_size (V), model.eos_id, model.pad_id
    prompt: a list of P >= 1 ints.
    cfg: a dict with the keys "canvas_length" (L), "max_canvases", "max_steps", "t_min",
        "t_max", "entropy_bound", "stability", "confidence".
    uniforms: (u_init, u_sample, u_noise), float32 in [0, 1), of shapes
        (max_canvases, L), (max_canvases, max_steps, L), (max_canvases, max_steps, L).
    Returns a list of ints: the prompt followed by L tokens per canvas generated.
    Nothing passed in is modified.
    """
    raise NotImplementedError
