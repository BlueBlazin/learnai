"""BlockDenoiser: a text-diffusion model that writes a block of tokens at a time.

A block text-diffusion model (DiffusionGemma) is autoregressive over *blocks*, not
tokens. The prompt is encoded once into a cache, as in any decoder. Then a *canvas* of
``L`` tokens starts as noise and is refined by a denoiser that sees the cached prefix
and the whole canvas at once (bidirectionally), over a handful of steps. The finished
canvas is appended to the text and encoded into the cache, and the next canvas starts.

    model = BlockDenoiser(ctx.fw, seed=3)                  # vocab 32; also sharpness=, eos_bias=
    cache = model.encode(prompt)                           # causal; counted
    logits = model.denoise(canvas, cache)                  # (L, vocab_size); counted
    logits = model.denoise(canvas, cache, previous_logits) # with self-conditioning
    cache = model.encode(finished_canvas, cache)           # extend the prefix by one block

Two calls, both functional (the cache passed in is never modified):

``encode(tokens, cache=None) -> DenoiserCache``
    Folds ``tokens`` (``(n,)``, ``n >= 1``) into the cache, causally and one token at a
    time, so encoding a prefix in any number of pieces gives a **bit-identical** cache.
    Counts one ``encoder_calls`` and ``n`` ``encoder_tokens``.

``denoise(canvas, cache, self_cond_logits=None) -> logits``
    Logits ``(L, vocab_size)`` for a canvas of ``L`` token ids placed right after the
    cached prefix. Position ``i`` reads the prefix, its own canvas token, every other
    canvas token (attention without a mask), and, if given, the logits of the previous
    step at the same position (``self_cond_logits``, ``(L, vocab_size)``: pass the
    temperature-scaled logits the previous step sampled from). Deterministic in its
    inputs. Counts one ``denoise_calls`` and ``L`` ``denoise_tokens``.

How it behaves, which is what fixtures are built from (measured with canvases of 8,
at most 12 steps, temperatures from 1.0 down to 0.2, entropy bound 1.0, and a stopping
rule of 2 stable steps and mean entropy below 0.5):

* The logits are ``sharpness * votes`` (plus ``eos_bias`` on ``eos_id``). The votes
  come from the prefix and the position, from the rest of the canvas (attention
  without a mask), from the canvas token itself (``OWN_VOTE`` for the token that is
  there), and from the previous step (``SELF_COND_VOTE`` times the softmax of
  ``self_cond_logits``). Accepted positions and repeated predictions therefore
  reinforce themselves, and leaving the self-conditioning out changes the result at
  every sharpness tried.
* ``sharpness`` 6 or more: every canvas stops after 3 or 4 steps, and at the stop
  every position is accepted, so the sampled, the noisy, and the argmax canvas are the
  same tokens. Use it for "stops early" fixtures.
* ``sharpness`` 0.5: the mean entropy never falls below 0.5, so every canvas runs to
  the step limit. Use it for "never converges" fixtures.
* ``sharpness`` 1 to 2: canvases settle after 3 to 12 steps, and the last step often
  still has unaccepted or mis-sampled positions, which is what separates the argmax
  canvas from the sampled one and from the noisy one. At 4 and above they coincide.
  The table of measurements is in ``docs/PROBLEM_AUTHORING.md`` (BlockDenoiser).
* ``eos_bias`` is added to the logit of ``eos_id`` at every position, after the
  scaling by ``sharpness``: raise it for a fixture that should end early, lower it for
  one that should fill every canvas. To rule the token out or in for certain it has
  to outweigh the votes, so use a large value (-100 never ends a text).

``pad_id`` is 0 and ``eos_id`` is 1, by the kit's convention. The arithmetic is numpy;
tensors at the boundary are the solution's. This is a toy with the interface and the
qualitative behavior of the shipped model, not a port of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from learnai.kit.counters import Counters

#: How much a canvas token raises its own logit, and how much probability 1 in the
#: self-conditioning logits raises a token's, both before ``sharpness`` scales them.
OWN_VOTE, SELF_COND_VOTE = 0.5, 2.0

_COUNTERS = ("encoder_calls", "encoder_tokens", "denoise_calls", "denoise_tokens")


@dataclass(frozen=True)
class DenoiserCache:
    """The encoded prefix: one row per token, and a running summary. Opaque, immutable."""

    rows: np.ndarray  # (length, d_model)
    summary: np.ndarray  # (d_model,)

    @property
    def length(self) -> int:
        """The number of tokens encoded so far."""
        return self.rows.shape[0]

    def __repr__(self) -> str:
        return f"DenoiserCache(length={self.length})"


class BlockDenoiser:
    """A seeded block-diffusion model. See the module docstring.

    Args:
        fw: the framework adapter (``ctx.fw``).
        seed: selects the weights.
        vocab_size, d_model, max_context: sizes.
        sharpness: scale of the prefix-and-position part of the logits (default 3.0).
        eos_bias: added to the logit of ``eos_id`` (default 0.0).

    Attributes:
        counters: ``encoder_calls``, ``encoder_tokens``, ``denoise_calls``,
            ``denoise_tokens``.
        vocab_size, pad_id (0), eos_id (1), sharpness, eos_bias, max_context.
    """

    pad_id = 0
    eos_id = 1

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        vocab_size: int = 32,
        d_model: int = 24,
        sharpness: float = 3.0,
        eos_bias: float = 0.0,
        max_context: int = 512,
    ) -> None:
        if vocab_size < 4 or d_model < 2 or d_model % 2:
            raise ValueError("vocab_size must be at least 4 and d_model even and at least 2")
        self.fw = fw
        self.seed = seed
        self.vocab_size, self.d_model = vocab_size, d_model
        self.sharpness, self.eos_bias = float(sharpness), float(eos_bias)
        self.max_context = max_context
        self.counters = Counters(*_COUNTERS)
        rng = np.random.default_rng([seed, 0x646966])
        d = d_model

        def normal(*shape: int, std: float) -> np.ndarray:
            array = (rng.standard_normal(shape) * std).astype(np.float32)
            array.setflags(write=False)
            return array

        self._w = {
            "embed": normal(vocab_size, d, std=1.0),
            "enc_in": normal(d, d, std=d**-0.5),
            "enc_rec": normal(d, d, std=d**-0.5),
            "ctx_q": normal(d, d, std=2.0 * d**-0.5),
            "ctx_k": normal(d, d, std=d**-0.5),
            "ctx_v": normal(d, d, std=d**-0.5),
            "summary": normal(d, d, std=d**-0.5),
            "position": normal(d, d, std=d**-0.5),
            "mix_q": normal(d, d, std=d**-0.5),
            "mix_k": normal(d, d, std=d**-0.5),
            "mix_v": normal(d, d, std=0.3 * d**-0.5),
            "head": normal(vocab_size, d, std=d**-0.5),
        }

    def with_framework(self, fw: Any) -> BlockDenoiser:
        """The same model speaking another framework, with fresh counters."""
        twin = object.__new__(BlockDenoiser)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters(*_COUNTERS)
        return twin

    # -- the two calls ----------------------------------------------------------------

    def encode(self, tokens: Any, cache: DenoiserCache | None = None) -> DenoiserCache:
        """Encode ``tokens`` after the prefix in ``cache`` (``None``: they start the text).

        Raises ``ValueError`` for an empty call, ids outside the vocabulary, a cache
        that is not this model's, or a total length beyond ``max_context``.
        """
        ids = self._ids(tokens, "tokens")
        cache = self._cache(cache, allow_none=True)
        if cache.length + ids.size > self.max_context:
            raise ValueError(
                f"sequence length {cache.length + ids.size} exceeds max_context {self.max_context}"
            )
        self.counters.add("encoder_calls")
        self.counters.add("encoder_tokens", int(ids.size))
        w = self._w
        rows, summary = [cache.rows], cache.summary
        # One token at a time, always: the cache is then the same bits however the
        # tokens were split over calls.
        for offset, token in enumerate(ids):
            x = w["embed"][token] + self._position(cache.length + offset)
            row = np.tanh(w["enc_in"] @ x + w["enc_rec"] @ summary).astype(np.float32)
            summary = (np.float32(0.85) * summary + np.float32(0.15) * row).astype(np.float32)
            rows.append(row[None])
        return self._frozen(np.concatenate(rows, axis=0), summary)

    def denoise(self, canvas: Any, cache: DenoiserCache, self_cond_logits: Any = None) -> Any:
        """Logits ``(L, vocab_size)`` for ``canvas`` (``(L,)`` ids) after the cached prefix.

        Raises ``ValueError`` for an empty canvas, ids outside the vocabulary, an empty
        or foreign cache (encode the prompt first), self-conditioning logits of another
        shape, or a canvas that would end beyond ``max_context``.
        """
        ids = self._ids(canvas, "canvas")
        cache = self._cache(cache, allow_none=False)
        if cache.length + ids.size > self.max_context:
            raise ValueError(
                f"the canvas ends at {cache.length + ids.size}, beyond max_context "
                f"{self.max_context}"
            )
        previous = None
        if self_cond_logits is not None:
            previous = np.asarray(
                self.fw.numpy(self_cond_logits)
                if self.fw.is_tensor(self_cond_logits)
                else self_cond_logits,
                dtype=np.float32,
            )
            if previous.shape != (ids.size, self.vocab_size):
                raise ValueError(
                    f"self_cond_logits must have shape ({ids.size}, {self.vocab_size}), got "
                    f"{previous.shape}"
                )
        self.counters.add("denoise_calls")
        self.counters.add("denoise_tokens", int(ids.size))
        return self.fw.tensor(self._denoise(ids, cache, previous))

    # -- arithmetic -------------------------------------------------------------------

    def _denoise(
        self, ids: np.ndarray, cache: DenoiserCache, previous: np.ndarray | None
    ) -> np.ndarray:
        w, d = self._w, self.d_model
        places = np.stack([self._position(cache.length + i) for i in range(ids.size)])
        # What the prefix says about each canvas position: attention over the cached
        # rows with a query made from the position, plus the running summary.
        scores = (places @ w["ctx_q"].T) @ (cache.rows @ w["ctx_k"].T).T / math.sqrt(d)
        context = _softmax(scores) @ (cache.rows @ w["ctx_v"].T)
        base = context + cache.summary @ w["summary"].T + places @ w["position"].T
        # What the rest of the canvas says: attention over all of it, no mask.
        tokens = w["embed"][ids] + places
        mixing = (tokens @ w["mix_q"].T) @ (tokens @ w["mix_k"].T).T / math.sqrt(d)
        hidden = base + _softmax(mixing) @ (tokens @ w["mix_v"].T)
        hidden = hidden / np.sqrt((hidden * hidden).mean(axis=-1, keepdims=True) + 1e-6)
        votes = hidden @ w["head"].T
        # A token already on the canvas votes for itself...
        votes[np.arange(ids.size), ids] += OWN_VOTE
        # ...and so does what the previous step believed (self-conditioning).
        if previous is not None:
            votes = votes + SELF_COND_VOTE * _softmax(previous)
        logits = self.sharpness * votes
        logits[:, self.eos_id] += self.eos_bias
        return logits.astype(np.float32)

    def _position(self, index: int) -> np.ndarray:
        half = self.d_model // 2
        angles = index * 50.0 ** (-np.arange(half, dtype=np.float64) / half)
        return np.concatenate((np.sin(angles), np.cos(angles))).astype(np.float32)

    # -- argument checks ----------------------------------------------------------------

    def _ids(self, tokens: Any, what: str) -> np.ndarray:
        ids = np.asarray(self.fw.numpy(tokens) if self.fw.is_tensor(tokens) else tokens)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError(f"{what} must have shape (n,) with n >= 1, got {ids.shape}")
        if ids.dtype.kind not in "iu":
            raise ValueError(f"{what} must be integers, got dtype {ids.dtype}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"{what} ids must be in [0, {self.vocab_size})")
        return ids.astype(np.int64)

    def _cache(self, cache: Any, *, allow_none: bool) -> DenoiserCache:
        if cache is None and allow_none:
            return self._frozen(
                np.zeros((0, self.d_model), np.float32), np.zeros(self.d_model, np.float32)
            )
        if not isinstance(cache, DenoiserCache) or cache.rows.shape[1:] != (self.d_model,):
            raise ValueError("cache must be a DenoiserCache returned by this model's encode")
        if cache.length == 0:
            raise ValueError("the cache is empty: encode the prompt first")
        return cache

    @staticmethod
    def _frozen(rows: np.ndarray, summary: np.ndarray) -> DenoiserCache:
        rows.setflags(write=False)
        summary.setflags(write=False)
        return DenoiserCache(rows, summary)

    def __repr__(self) -> str:
        return (
            f"BlockDenoiser(seed={self.seed}, vocab_size={self.vocab_size}, "
            f"sharpness={self.sharpness:g}, eos_bias={self.eos_bias:g})"
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
