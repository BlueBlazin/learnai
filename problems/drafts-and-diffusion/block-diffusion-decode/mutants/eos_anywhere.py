# BUG: the end-of-sequence token is looked for in the whole text, so one in the prompt ends generation after the first canvas.
from typing import Any

import torch

from learnai.kit import ops


def diffusion_generate(
    model: Any,
    prompt: list[int],
    cfg: dict[str, Any],
    uniforms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[int]:
    u_init, u_sample, u_noise = uniforms
    length, steps, vocab = cfg["canvas_length"], cfg["max_steps"], model.vocab_size
    out = list(prompt)
    cache = model.encode(prompt)  # the prompt goes through the encoder once

    for c in range(cfg["max_canvases"]):
        canvas = torch.floor(u_init[c] * vocab).long()  # a canvas starts as noise
        previous = None  # self-conditioning starts over with every canvas
        drafts = []  # the argmax canvases of this canvas's steps
        for j in range(steps):
            remaining = steps - j  # counts this step: the first step is the hottest
            temperature = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * remaining / steps
            scaled = model.denoise(canvas, cache, previous) / temperature

            # The step of entropy-bound-denoise-step, on the scaled logits.
            log_p = torch.log_softmax(scaled, dim=-1)
            p = torch.exp(log_p)
            cdf = torch.cumsum(p, dim=-1)
            sampled = (cdf <= u_sample[c, j][:, None] * cdf[:, -1:]).sum(dim=-1)
            accepted = ops.eb_accept(scaled, cfg["entropy_bound"])
            noise = torch.floor(u_noise[c, j] * vocab).long()
            canvas = torch.where(accepted, sampled, noise)  # the next step's input
            draft = torch.argmax(scaled, dim=-1)

            # Stop when the draft repeated the last `stability` drafts and the
            # distributions are sharp. The slice is empty when stability is 0.
            recent = drafts[len(drafts) - cfg["stability"] :]
            stable = len(drafts) >= cfg["stability"] and all(
                torch.equal(draft, old) for old in recent
            )
            confident = float(-(p * log_p).sum(dim=-1).mean()) < cfg["confidence"]
            drafts.append(draft)
            previous = scaled  # what the next step is conditioned on
            if stable and confident:
                break

        # The text gets the draft of the last step, never the sampled canvas.
        final = draft.tolist()
        if model.eos_id in out + final:
            cut = final.index(model.eos_id) + 1 if model.eos_id in final else length
            return out + final[:cut] + [model.pad_id] * (length - cut)
        out += final
        if c + 1 < cfg["max_canvases"]:  # the last canvas is never read again
            cache = model.encode(draft, cache)
    return out
