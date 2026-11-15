"""RecurrentLM: a language model whose memory is a fixed-size state, not a KV cache.

A transformer remembers a prefix as keys and values, one entry per token: to forget
the last three tokens, drop three entries (``KVCache.truncate``). A recurrent layer
(a state-space layer such as Mamba, a Gated DeltaNet layer in a hybrid model) folds
every token into a state of fixed size. The state after 40 tokens is as large as the
state after 4, and there is no operation that takes the last tokens back out. Reusing
a cached prefix therefore needs a *saved copy of the state at that prefix*: a
checkpoint. That is what this model is for.

    lm = RecurrentLM(ctx.fw, seed=3)
    logits, state = lm(tokens)               # no state: tokens are the whole prefix
    logits, state = lm(more_tokens, state)   # with state: tokens continue it
    state.length                              # tokens folded into it; there is no truncate

Same call convention as ``TinyLM``: ``tokens`` is ``(T,)`` or ``(B, T)``, ``logits``
has one row per token of this call, and the state passed in is never modified. A
``RecurrentState`` is opaque and immutable; keep any number of them (a dict from
prefix length to state is a checkpoint store). ``state.nbytes`` is its size, the same
at every length.

Because every token is folded in by the same arithmetic one at a time, continuing
from a saved state gives **bit-identical** logits to reading the whole prefix in one
call. Tests may therefore compare logits exactly, which they cannot with ``TinyLM``.

Counters: ``forward_calls`` and ``tokens_processed`` (batch size times length, summed
over calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters
from whetstone.kit.tinylm import LOGIT_STD, _rms_norm


@dataclass(frozen=True)
class RecurrentState:
    """The model's memory of a prefix: one ``(batch, d_state)`` array per layer.

    Fixed-size, opaque, immutable. ``length`` is the number of tokens it has absorbed.
    It cannot be truncated: to go back, keep the earlier state.
    """

    layers: tuple[np.ndarray, ...]
    length: int
    batched: bool

    @property
    def batch_size(self) -> int:
        return self.layers[0].shape[0]

    @property
    def nbytes(self) -> int:
        """Bytes this state occupies: the same whatever its ``length``."""
        return sum(layer.nbytes for layer in self.layers)

    def select(self, rows: Any) -> RecurrentState:
        """Batch rows ``rows`` of this state, in that order (as ``KVCache.select``)."""
        index = np.asarray(rows.tolist() if hasattr(rows, "tolist") else rows)
        if index.ndim != 1 or index.size == 0 or index.dtype.kind not in "iu":
            raise ValueError("rows must be a non-empty 1-D sequence of ints")
        if index.min() < 0 or index.max() >= self.batch_size:
            raise ValueError(f"rows must be in [0, {self.batch_size}), got {index.tolist()}")
        layers = tuple(layer[index] for layer in self.layers)
        for layer in layers:
            layer.setflags(write=False)
        return RecurrentState(layers, self.length, True)

    def truncate(self, n: int) -> RecurrentState:
        """Always raises: a recurrent state cannot give tokens back."""
        raise TypeError(
            "a RecurrentState cannot be truncated: its tokens are folded into a fixed-size "
            "state. Keep the state you had after the prefix you want to return to."
        )

    def __repr__(self) -> str:
        return f"RecurrentState(length={self.length}, layers={len(self.layers)})"


class RecurrentLM:
    """A seeded recurrent language model. See the module docstring.

    Args:
        fw: the framework adapter (``ctx.fw``).
        seed: selects the weights.
        vocab_size, d_model, d_state, n_layers, max_context: sizes.

    Each layer keeps a state ``s`` and, per token with input ``x``, computes

        f = sigmoid(norm(x) W_f)                 a forget gate per state element
        s = f * s + (1 - f) * tanh(norm(x) W_in)
        x = x + s W_out

    Attributes ``bos_id`` (0) and ``eos_id`` (1) as on ``TinyLM``.
    """

    bos_id = 0
    eos_id = 1

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        vocab_size: int = 64,
        d_model: int = 32,
        d_state: int = 48,
        n_layers: int = 2,
        max_context: int = 256,
    ) -> None:
        self.fw = fw
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.max_context = max_context
        self.counters = Counters("forward_calls", "tokens_processed")
        rng = np.random.default_rng([seed, 0x726E6E])

        def normal(*shape: int, std: float) -> np.ndarray:
            return (rng.standard_normal(shape) * std).astype(np.float32)

        d, s = d_model, d_state
        self._weights = {
            "tok": normal(vocab_size, d, std=1.0),
            "layers": [
                {
                    "norm": np.ones(d, dtype=np.float32),
                    "w_f": normal(d, s, std=d**-0.5),
                    # A bias toward remembering: without it the state forgets a token
                    # within a step or two and long prefixes would not matter.
                    "b_f": np.full(s, 2.5, dtype=np.float32),
                    "w_in": normal(d, s, std=d**-0.5),
                    "w_out": normal(s, d, std=s**-0.5),
                }
                for _ in range(n_layers)
            ],
            "norm_out": np.ones(d, dtype=np.float32),
            "head": normal(d, vocab_size, std=LOGIT_STD * d**-0.5),
        }

    def with_framework(self, fw: Any) -> RecurrentLM:
        """The same model speaking another framework, with fresh counters."""
        twin = object.__new__(RecurrentLM)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("forward_calls", "tokens_processed")
        return twin

    def initial_state(self, batch_size: int | None = None) -> RecurrentState:
        """The state before any token (all zeros): what a call without a state starts from."""
        rows = 1 if batch_size is None else batch_size
        layers = tuple(
            np.zeros((rows, self.d_state), dtype=np.float32) for _ in range(self.n_layers)
        )
        for layer in layers:
            layer.setflags(write=False)
        return RecurrentState(layers, 0, batch_size is not None)

    def __call__(
        self, tokens: Any, state: RecurrentState | None = None
    ) -> tuple[Any, RecurrentState]:
        """Run the model on ``tokens``, continuing from ``state`` if given.

        Returns ``(logits, new_state)``. Raises ``ValueError`` for token ids outside
        the vocabulary, an empty call, a batch size that does not match the state, or
        a total length beyond ``max_context``.
        """
        ids = np.asarray(self.fw.numpy(tokens) if self.fw.is_tensor(tokens) else tokens)
        if ids.ndim not in (1, 2) or ids.size == 0:
            raise ValueError(f"tokens must have shape (T,) or (B, T) with T >= 1, got {ids.shape}")
        if ids.dtype.kind not in "iu":
            raise ValueError(f"tokens must be integers, got dtype {ids.dtype}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        batched = ids.ndim == 2
        ids = ids if batched else ids[None]
        batch, new = ids.shape
        if state is None:
            state = self.initial_state(batch if batched else None)
        if not isinstance(state, RecurrentState) or len(state.layers) != self.n_layers:
            raise ValueError("state must be a RecurrentState of this model")
        if state.batch_size != batch:
            raise ValueError(f"batch size {batch} does not match the state's {state.batch_size}")
        if state.length + new > self.max_context:
            raise ValueError(
                f"sequence length {state.length + new} exceeds max_context {self.max_context}"
            )
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", batch * new)

        w = self._weights
        memory = [layer.copy() for layer in state.layers]
        logits = np.empty((batch, new, self.vocab_size), dtype=np.float32)
        # One token at a time, always: the result of a call is then the same bits
        # however the tokens are split over calls.
        for t in range(new):
            x = w["tok"][ids[:, t]]
            for index, layer in enumerate(w["layers"]):
                h = _rms_norm(x, layer["norm"])
                forget = 1.0 / (1.0 + np.exp(-(h @ layer["w_f"] + layer["b_f"])))
                memory[index] = (
                    forget * memory[index] + (1.0 - forget) * np.tanh(h @ layer["w_in"])
                ).astype(np.float32)
                x = (x + memory[index] @ layer["w_out"]).astype(np.float32)
            logits[:, t] = _rms_norm(x, w["norm_out"]) @ w["head"]
        for layer in memory:
            layer.setflags(write=False)
        new_state = RecurrentState(tuple(memory), state.length + new, batched or state.batched)
        return self.fw.tensor(logits if batched else logits[0]), new_state

    def __repr__(self) -> str:
        return (
            f"RecurrentLM(seed={self.seed}, vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"d_state={self.d_state}, n_layers={self.n_layers}, max_context={self.max_context})"
        )
