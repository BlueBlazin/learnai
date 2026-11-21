"""Regenerate the fixture of block-diffusion-decode.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/drafts-and-diffusion/block-diffusion-decode/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. For every
scenario of ``tests.py`` (``SCENARIOS``: a toy ``BlockDenoiser``, a prompt, a
configuration, and the uniforms the tests draw for it) this runs
``DiffusionGemmaGenerationMixin.generate`` of
``models/diffusion_gemma/generation_diffusion_gemma.py`` and records the tokens it
returns, the sizes of its encoder calls, and its number of decoder calls. The tests
require their own loop, and through it every solution, to agree with all three.

What is the library's: the canvas loop and its bookkeeping (what is encoded when, what
is appended, when it ends), ``LinearTemperatureScheduleLogitsProcessor``, the softmax
and the argmax of ``_denoising_step``, ``EntropyBoundSampler.accept_canvas`` and
``renoise_canvas``, ``StableAndConfidentStoppingCriteria``, the self-conditioning
hand-over, ``EosTokenCriteria``, and the padding of ``_finalize_canvas``.

What is supplied from here, because the library takes it from a model or a generator:

- the encoder and the decoder are the toy ``BlockDenoiser`` (``Toy.encode`` and
  ``Toy.forward``), and the cache object carries the toy's ``DenoiserCache``; the
  library's attention masks are not built. Nothing about the shipped *model* is
  compared, only the loop around it;
- ``torch.multinomial``, which samples the canvas, is replaced by inverse-CDF sampling
  from ``u_sample`` with the rule of ``categorical-sample``, applied to the
  probabilities the step passes it;
- ``EntropyBoundSampler.initialize_canvas`` (``torch.randint``), which the library calls
  once when a canvas starts and once per step for the noise, returns
  ``floor(u_init * V)`` and ``floor(u_noise * V)``.

``max_new_tokens`` is ``max_canvases * canvas_length``, a batch of one sequence.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import transformers
from transformers.models.diffusion_gemma import generation_diffusion_gemma as dg

from learnai.judge import frameworks

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("block_diffusion_tests", HERE.parent / "tests.py")
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)


class Prefix:
    """Stands in for the library's key-value cache: it holds the toy's DenoiserCache."""

    is_compileable = False

    def __init__(self, cache=None):
        self.cache = cache

    def get_seq_length(self):
        return 0 if self.cache is None else self.cache.length


class Sampler(dg.EntropyBoundSampler):
    """The library's sampler with its one random draw replaced."""

    feed = None  # set per run: returns the next canvas of noise

    def initialize_canvas(self, batch_size, device):
        return Sampler.feed()


class Toy(dg.DiffusionGemmaGenerationMixin):
    """The least a model needs for the library's generate(), around a BlockDenoiser."""

    def __init__(self, denoiser, length):
        self.denoiser = denoiser
        self.config = SimpleNamespace(
            canvas_length=length,
            image_token_id=-1,
            text_config=SimpleNamespace(vocab_size=denoiser.vocab_size),
        )
        self.generation_config = dg.DiffusionGemmaGenerationConfig()
        self.dtype = torch.float32
        encoder = self.encode
        encoder.__dict__["create_masks_for_generate"] = lambda **kwargs: None
        self.model = SimpleNamespace(
            encoder=encoder,
            decoder=SimpleNamespace(
                create_diffusion_decoder_attention_mask=lambda **kwargs: None,
                embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
            ),
        )
        self.encoded, self.denoised = [], 0

    def encode(self, input_ids, attention_mask, past_key_values, position_ids, **kwargs):
        assert input_ids.shape[0] == 1
        self.encoded.append(input_ids.shape[1])
        longer = self.denoiser.encode(input_ids[0], past_key_values.cache)
        return SimpleNamespace(past_key_values=Prefix(longer))

    def forward(self, decoder_input_ids, self_conditioning_logits, past_key_values, **kwargs):
        self.denoised += 1
        previous = None if self_conditioning_logits is None else self_conditioning_logits[0]
        logits = self.denoiser.denoise(decoder_input_ids[0], past_key_values.cache, previous)
        return SimpleNamespace(logits=logits[None])

    def _prepare_sampler(self, generation_config):
        return Sampler(
            generation_config.sampler_config,
            self.config.canvas_length,
            self.config.text_config.vocab_size,
            generation_config.max_denoising_steps,
        )


def shipped_generate(denoiser, prompt, cfg, uniforms):
    """What the library's loop does: tokens, encoder call sizes, decoder calls."""
    u_init, u_sample, u_noise = (torch.from_numpy(u) for u in uniforms)
    vocab = denoiser.vocab_size
    toy = Toy(denoiser, cfg["canvas_length"])
    at = {"canvas": -1, "step": -1}

    def feed():
        canvas = len(toy.encoded) - 1  # the encoder has run once per canvas started
        if canvas != at["canvas"]:  # the first call of a canvas: its starting noise
            at["canvas"], at["step"] = canvas, -1
            return torch.floor(u_init[canvas] * vocab).long()[None]
        return torch.floor(u_noise[canvas, at["step"]] * vocab).long()[None]

    def inverse_cdf(probs, num_samples):
        assert num_samples == 1
        at["step"] += 1
        cdf = torch.cumsum(probs, dim=-1)
        u = u_sample[at["canvas"], at["step"]]
        return (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1, keepdim=True)

    Sampler.feed = staticmethod(feed)
    config = dg.DiffusionGemmaGenerationConfig(
        max_new_tokens=cfg["max_canvases"] * cfg["canvas_length"],
        max_denoising_steps=cfg["max_steps"],
        sampler_config=dg.EntropyBoundSamplerConfig(entropy_bound=float(cfg["entropy_bound"])),
        t_min=cfg["t_min"],
        t_max=cfg["t_max"],
        stability_threshold=cfg["stability"],
        confidence_threshold=float(cfg["confidence"]),
        eos_token_id=denoiser.eos_id,
        pad_token_id=denoiser.pad_id,
    )
    with mock.patch.object(torch, "multinomial", side_effect=inverse_cdf), torch.no_grad():
        out = toy.generate(
            torch.tensor([list(prompt)]), past_key_values=Prefix(), generation_config=config
        )
    return {
        "tokens": out.sequences[0].tolist(),
        "encoder_calls": toy.encoded,
        "denoise_calls": toy.denoised,
    }


def main() -> None:
    defaults = dg.DiffusionGemmaGenerationConfig._get_default_generation_params()
    shipped = tests.SHIPPED
    assert (defaults["t_min"], defaults["t_max"]) == (shipped["t_min"], shipped["t_max"])
    assert defaults["sampler_config"].entropy_bound == shipped["entropy_bound"]
    assert defaults["stability_threshold"] == shipped["stability"]
    assert defaults["confidence_threshold"] == shipped["confidence"]
    assert defaults["max_denoising_steps"] == 48  # the tests use 12 and fewer

    ctx = SimpleNamespace(fw=frameworks.get("torch"))
    out = {}
    for name, kwargs in tests.SCENARIOS.items():
        made = tests.Scenario(ctx, **kwargs)
        out[name] = shipped_generate(made.maker(), made.prompt, made.cfg, made.uniforms)
        agree = out[name]["tokens"] == made.want["tokens"] and out[name]["denoise_calls"] == sum(
            made.want["steps"]
        )
        print(
            f"{name}: steps {made.want['steps']}, encoder calls {out[name]['encoder_calls']}, "
            f"{'agrees with' if agree else 'DIFFERS from'} the tests' loop"
        )
    (HERE / "shipped.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote shipped.json ({len(out)} scenarios, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
