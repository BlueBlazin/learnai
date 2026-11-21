"""Regenerate the fixture of entropy-bound-denoise-step.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/drafts-and-diffusion/entropy-bound-denoise-step/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. Every expected
array is computed by ``DiffusionGemmaGenerationMixin._denoising_step`` of
``models/diffusion_gemma/generation_diffusion_gemma.py`` with its own
``LinearTemperatureScheduleLogitsProcessor`` and ``EntropyBoundSampler``, so the tests
compare a solution with the shipped code and not only with this problem's reference.

``_denoising_step`` is called as a plain function. Three things are supplied from here:

- the decoder: a function that returns fixed ``logits`` whatever it is given;
- ``torch.multinomial``, which the step calls to sample one token per position, is
  replaced for the duration of the call by inverse-CDF sampling from ``u_sample`` with
  the rule of the ``categorical-sample`` problem, applied to the probabilities the step
  passes it;
- ``EntropyBoundSampler.initialize_canvas``, which the sampler calls for the noise
  tokens (``torch.randint``), is replaced by ``floor(u_noise * V)``.

The softmax, the temperature, the entropies, the sort, the acceptance mask, the
renoising, and the argmax canvas are the library's. No stopping criterion is passed.

Schedule values are those of the ``generation_config.json`` of
``google/diffusiongemma-26B-A4B-it`` (read on 2026-11-21) and of the library's
defaults: ``t_min`` 0.4, ``t_max`` 0.8, ``max_denoising_steps`` 48, ``entropy_bound``
0.1. Per run ``r{n}``: ``current``, ``logits``, ``n`` (steps remaining), ``n_max``,
``t_min``, ``t_max``, ``entropy_bound``, ``u_sample``, ``u_noise`` -> ``temperature``
(``logits / processed logits``, as float64), ``canvas``, ``accepted``, ``argmax``.

Inputs are built so that no decision sits on a float32 rounding boundary (the
statement's Constraints): every ``u_sample`` is at least 1e-3 away from each cumulative
probability, every ``u_noise * V`` is the middle of an integer interval, and each bound
is at least 1e-3 away from the sums it is compared with, between two positions whose
entropies differ by at least 1e-3.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.diffusion_gemma import generation_diffusion_gemma as dg

HERE = Path(__file__).parent
T_MIN, T_MAX, N_MAX, BOUND = 0.4, 0.8, 48, 0.1


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def entropies(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)


def safe_uniforms(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniform per row in the middle of a token's interval at least 4e-3 wide."""
    cdf = np.cumsum(p, axis=-1)
    low = np.concatenate((np.zeros((p.shape[0], 1)), cdf[:, :-1]), axis=-1)
    out = np.empty(p.shape[0], dtype=np.float32)
    for i in range(p.shape[0]):
        token = int(np.searchsorted(cdf[i], rng.random(), side="right"))
        token = min(token, p.shape[1] - 1)
        if p[i, token] < 4e-3:
            token = int(p[i].argmax())
        out[i] = low[i, token] + p[i, token] * rng.uniform(0.3, 0.7)
    return out


def noise_uniforms(length: int, vocab: int, rng: np.random.Generator) -> np.ndarray:
    return ((rng.integers(0, vocab, length) + rng.uniform(0.25, 0.75, length)) / vocab).astype(
        np.float32
    )


def bound_accepting(h: np.ndarray, count: int) -> float:
    """A bound that accepts exactly `count` positions, clear of every rounding question."""
    ranked = np.sort(h)
    last, first_out = ranked[count - 1], ranked[count]
    assert first_out - last >= 1e-3 and last >= 2e-3, (last, first_out)
    return float(ranked[: count - 1].sum() + last / 2)


def shipped_step(current, logits, n, n_max, t_min, t_max, entropy_bound, u_sample, u_noise):
    length, vocab = logits.shape
    fake = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab)),
        model=SimpleNamespace(
            decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1)))
        ),
    )
    raw = torch.from_numpy(logits)[None]
    config = dg.EntropyBoundSamplerConfig(entropy_bound)
    sampler = dg.EntropyBoundSampler(config, length, vocab, n_max)
    processors = LogitsProcessorList(
        [dg.LinearTemperatureScheduleLogitsProcessor(t_min, t_max, n_max)]
    )
    uniforms = torch.from_numpy(u_sample)

    def inverse_cdf(probs, num_samples):
        assert num_samples == 1 and probs.shape == (length, vocab)
        cdf = torch.cumsum(probs, dim=-1)
        return (cdf <= uniforms[:, None] * cdf[:, -1:]).sum(dim=-1, keepdim=True)

    noise = torch.floor(torch.from_numpy(u_noise) * vocab).long()[None]
    with (
        mock.patch.object(torch, "multinomial", side_effect=inverse_cdf) as sample,
        mock.patch.object(sampler, "initialize_canvas", return_value=noise) as renoise,
    ):
        canvas, argmax, processed, _ = dg.DiffusionGemmaGenerationMixin._denoising_step(
            fake,
            decoder_forward=lambda **kwargs: SimpleNamespace(logits=raw),
            current_canvas=torch.from_numpy(current)[None],
            argmax_canvas=torch.from_numpy(current)[None],
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            decoder_position_ids=None,
            self_conditioning_logits=None,
            mask_mapping=None,
            past_key_values=None,
            finished_denoising=torch.zeros(1, dtype=torch.bool),
            cur_step=n,
            sampler=sampler,
            logits_processor=processors,
            diffusion_stopping_criteria=None,
        )
    assert sample.call_count == 1 and renoise.call_count == 1
    ratio = (raw.double() / processed.double())[raw != 0]
    return {
        "temperature": np.array(float(ratio.mean())),
        "canvas": canvas[0].numpy(),
        "accepted": sampler.accepted_token_mask[0].numpy(),
        "argmax": argmax[0].numpy(),
    }


def main() -> None:
    assert (
        dg.DiffusionGemmaGenerationConfig._get_default_generation_params()[
            "sampler_config"
        ].entropy_bound
        == BOUND
    )
    defaults = dg.DiffusionGemmaGenerationConfig._get_default_generation_params()
    assert (defaults["t_min"], defaults["t_max"], defaults["max_denoising_steps"]) == (
        T_MIN,
        T_MAX,
        N_MAX,
    )
    rng = np.random.default_rng(2407)
    runs = []

    # Run 0: the statement's worked example, then the same inputs with a smaller bound.
    ln = np.log
    example = np.array(
        [[0, 0, 0.5 * ln(4 / 3)], [0.5 * ln(18), 0, 0], [0, 0.5 * ln(2), 0], [0.5 * ln(18), 0, 0]],
        dtype=np.float32,
    )
    for bound in (0.5, 0.3):
        runs.append(
            {
                "current": np.array([0, 1, 2, 0]),
                "logits": example,
                "n": 12,  # 12 of 48 steps remaining: temperature 0.5
                "n_max": N_MAX,
                "t_min": T_MIN,
                "t_max": T_MAX,
                "entropy_bound": bound,
                "u_sample": np.array([0.5, 0.5, 0.7, 0.97], dtype=np.float32),
                "u_noise": np.array([0.9, 0.1, 0.4, 0.5], dtype=np.float32),
            }
        )

    # Runs 2 to 7: 16 positions, 40 tokens, sharpness from nearly certain to nearly
    # uniform, at the shipped schedule; the bound accepts a chosen number of positions.
    for n, count in ((48, 3), (30, 7), (11, 12), (1, 5), (48, 1), (2, 15)):
        length, vocab = 16, 40
        temperature = T_MIN + (T_MAX - T_MIN) * n / N_MAX
        while True:
            sharp = rng.uniform(0.1, 4.0, (length, 1))
            logits = (rng.standard_normal((length, vocab)) * sharp).astype(np.float32)
            p = probabilities(logits, temperature)
            ranked = np.sort(entropies(p))
            top2 = np.sort(logits, axis=-1)[:, -2:]
            if (
                ranked[count] - ranked[count - 1] >= 1e-3
                and ranked[count - 1] >= 2e-3
                and (top2[:, 1] - top2[:, 0]).min() > 1e-3
            ):
                break
        runs.append(
            {
                "current": rng.integers(0, vocab, length),
                "logits": logits,
                "n": n,
                "n_max": N_MAX,
                "t_min": T_MIN,
                "t_max": T_MAX,
                "entropy_bound": bound_accepting(entropies(p), count),
                "u_sample": safe_uniforms(p, rng),
                "u_noise": noise_uniforms(length, vocab, rng),
            }
        )

    # Run 8: the shipped bound 0.1 late in denoising: most positions are certain (their
    # probabilities underflow to exactly 0 and 1 in float32), a few are not.
    length, vocab, n = 16, 40, 3
    temperature = T_MIN + (T_MAX - T_MIN) * n / N_MAX
    logits = (rng.standard_normal((length, vocab)) * 2.0).astype(np.float32)
    certain = rng.permutation(length)[:11]
    logits[certain, rng.integers(0, vocab, 11)] += 300.0
    p = probabilities(logits, temperature)
    h = entropies(p)
    ranked = np.sort(h)
    sums = np.cumsum(ranked) - ranked
    # The 11 certain positions and the least uncertain of the others are accepted.
    assert np.abs(sums - BOUND).min() > 1e-3 and (sums <= BOUND).sum() == 12
    runs.append(
        {
            "current": rng.integers(0, vocab, length),
            "logits": logits,
            "n": n,
            "n_max": N_MAX,
            "t_min": T_MIN,
            "t_max": T_MAX,
            "entropy_bound": BOUND,
            "u_sample": safe_uniforms(p, rng),
            "u_noise": noise_uniforms(length, vocab, rng),
        }
    )

    out = {
        "transformers_version": np.array(transformers.__version__),
        "runs": np.array(len(runs)),
    }
    for i, run in enumerate(runs):
        result = shipped_step(**run)
        for name, value in {**run, **result}.items():
            out[f"r{i}_{name}"] = np.asarray(value)
        print(
            f"run {i}: temperature {float(result['temperature']):.6f}, "
            f"accepted {int(result['accepted'].sum())} of {len(result['accepted'])}"
        )
    np.savez(HERE / "shipped.npz", **out)
    print(f"wrote shipped.npz ({len(runs)} runs, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
