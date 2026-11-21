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
    u_init, u_sample, u_noise = uniforms
    length, steps, vocab = cfg["canvas_length"], cfg["max_steps"], model.vocab_size
    out = list(prompt)
    cache = model.encode(prompt)  # the prompt goes through the encoder once

    for c in range(cfg["max_canvases"]):
        canvas = jnp.floor(u_init[c] * vocab).astype(jnp.int32)  # a canvas starts as noise
        previous = None  # self-conditioning starts over with every canvas
        drafts = []  # the argmax canvases of this canvas's steps
        for j in range(steps):
            remaining = steps - j  # counts this step: the first step is the hottest
            temperature = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * remaining / steps
            scaled = model.denoise(canvas, cache, previous) / temperature

            # The step of entropy-bound-denoise-step, on the scaled logits.
            log_p = jax.nn.log_softmax(scaled, axis=-1)
            p = jnp.exp(log_p)
            cdf = jnp.cumsum(p, axis=-1)
            sampled = (cdf <= u_sample[c, j][:, None] * cdf[:, -1:]).sum(axis=-1)
            accepted = ops.eb_accept(scaled, cfg["entropy_bound"])
            noise = jnp.floor(u_noise[c, j] * vocab).astype(jnp.int32)
            canvas = jnp.where(accepted, sampled, noise)  # the next step's input
            draft = jnp.argmax(scaled, axis=-1)

            # Stop when the draft repeated the last `stability` drafts and the
            # distributions are sharp. The slice is empty when stability is 0.
            recent = drafts[len(drafts) - cfg["stability"] :]
            stable = len(drafts) >= cfg["stability"] and all(
                bool(jnp.array_equal(draft, old)) for old in recent
            )
            confident = float(-(p * log_p).sum(axis=-1).mean()) < cfg["confidence"]
            drafts.append(draft)
            previous = scaled  # what the next step is conditioned on
            if stable and confident:
                break

        # The text gets the draft of the last step, never the sampled canvas.
        final = draft.tolist()
        if model.eos_id in final:
            cut = final.index(model.eos_id) + 1
            return out + final[:cut] + [model.pad_id] * (length - cut)
        out += final
        if c + 1 < cfg["max_canvases"]:  # the last canvas is never read again
            cache = model.encode(draft, cache)
    return out
