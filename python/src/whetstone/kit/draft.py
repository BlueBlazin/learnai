"""draft_target_pair: two TinyLMs that agree most of the time, for speculative decoding.

Speculative decoding runs a cheap *draft* model ahead and lets the expensive *target*
model check its guesses. How much that saves depends on how often the two agree, so a
problem needs a pair with a known agreement rate:

    draft, target = draft_target_pair(fw, seed=3, agreement=0.8)

``target`` is ``TinyLM(fw, seed)``. ``draft`` has the same shape and the same weights
plus seeded noise, and the amount of noise is tuned so that, along the target's own
greedy continuations, the draft's greedy token equals the target's at about the
requested rate. Each model has its own counters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from whetstone.kit.tinylm import TinyLM

#: The calibration set: this many seeded random prompts of this length, each continued
#: greedily by the target for this many tokens.
_PROMPTS, _PROMPT_LEN, _CONTINUATION = 24, 6, 26
#: Noise levels searched, as a multiple of each weight matrix's own standard deviation.
_MAX_NOISE = 4.0
_BISECTION_STEPS = 24


def draft_target_pair(
    fw: Any, seed: int = 0, agreement: float = 0.8, **model_args: Any
) -> tuple[TinyLM, TinyLM]:
    """Return ``(draft, target)``: two models over one vocabulary that mostly agree.

    Args:
        fw: the framework adapter both models speak (``ctx.fw``).
        seed: selects the target's weights, the noise, and the calibration prompts.
        agreement: the fraction of positions, in ``(0, 1]``, at which the draft's
            greedy token should equal the target's. ``1.0`` gives a draft with exactly
            the target's weights.
        **model_args: passed to ``TinyLM`` (``vocab_size``, ``d_model``, ...).

    The rate is met closely on the calibration prompts and to within a few percent on
    other prompts (it is a statistical property of the pair, not a guarantee for one
    prompt). The result is a pure function of the arguments.
    """
    if not 0.0 < agreement <= 1.0:
        raise ValueError(f"agreement must be in (0, 1], got {agreement}")
    target = TinyLM(fw, seed, **model_args)
    # Calibrate on a numpy-speaking twin: no tensors of `fw` are created, and the
    # models handed back have done no work.
    plain = target.with_framework(_Numpy())
    context, wanted = _calibration_set(plain, seed)

    rng = np.random.default_rng([seed, 1])
    noise = _noise_like(plain._weights, rng)

    def rate(scale: float) -> float:
        return float(np.mean(_greedy(_perturbed(plain, noise, scale), context) == wanted))

    # More noise means less agreement (up to sampling wobble), so bisect on the noise
    # scale, keeping `low` on the side that agrees at least as often as requested.
    low, high = 0.0, _MAX_NOISE
    if agreement < 1.0:
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            if rate(middle) >= agreement:
                low = middle
            else:
                high = middle
    return _perturbed(plain, noise, low).with_framework(fw), target


class _Numpy:
    """A do-nothing adapter: the model's numpy arrays pass straight through."""

    def is_tensor(self, value: Any) -> bool:
        return False

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        return value


def _calibration_set(target: TinyLM, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Contexts ``(N, T)`` on the target's greedy paths, and its greedy token after each.

    Each row is a random prompt followed by the target's own greedy continuation. Only
    positions from the last prompt token on are returned: those are the contexts a
    speculative decoder actually asks both models about.
    """
    rng = np.random.default_rng([seed, 0])
    length = min(_PROMPT_LEN + _CONTINUATION, target.max_context)
    prompt_len = min(_PROMPT_LEN, length - 1)
    tokens = rng.integers(0, target.vocab_size, size=(_PROMPTS, prompt_len))
    logits, cache = target(tokens)
    for _ in range(length - prompt_len):
        nxt = logits[:, -1].argmax(axis=-1)[:, None]
        tokens = np.concatenate([tokens, nxt], axis=1)
        logits, cache = target(nxt, cache)
    return tokens, _greedy(target, tokens)


def _greedy(lm: TinyLM, context: np.ndarray) -> np.ndarray:
    """The model's greedy next token after each position from the last prompt token on."""
    logits, _ = lm(context)
    return logits.argmax(axis=-1)[:, min(_PROMPT_LEN, context.shape[1]) - 1 :]


def _noise_like(weights: Any, rng: np.random.Generator) -> Any:
    """Standard normal noise shaped like every weight matrix; None for the norm gains."""
    if isinstance(weights, dict):
        return {name: _noise_like(value, rng) for name, value in weights.items()}
    if isinstance(weights, list):
        return [_noise_like(value, rng) for value in weights]
    if weights.ndim < 2:
        return None
    return rng.standard_normal(weights.shape).astype(np.float32)


def _add_noise(weights: Any, noise: Any, scale: float) -> Any:
    if isinstance(weights, dict):
        return {name: _add_noise(value, noise[name], scale) for name, value in weights.items()}
    if isinstance(weights, list):
        return [_add_noise(value, n, scale) for value, n in zip(weights, noise, strict=True)]
    if noise is None:
        return weights
    # Noise relative to the matrix's own spread, so every matrix is disturbed alike.
    return (weights + np.float32(scale * weights.std()) * noise).astype(np.float32)


def _perturbed(target: TinyLM, noise: Any, scale: float) -> TinyLM:
    """A twin of ``target`` whose weights are disturbed by ``scale`` times the noise."""
    draft = target.with_framework(target.fw)
    draft._weights = _add_noise(target._weights, noise, scale)
    return draft
