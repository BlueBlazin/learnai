"""TinyLM: a tiny, deterministic, decoder-only transformer for practice problems.

Problems hand a ``TinyLM`` to your solution the way LeetCode hands you a ``ListNode``.
It is a real causal language model (embeddings, pre-norm attention blocks, an MLP, an
output head), only very small, with weights drawn from a seeded generator instead
of training. Its outputs are meaningless as language but fully deterministic, which is
what a judge needs.

The arithmetic is plain numpy, written once, so every framework sees bit-identical
logits. At the boundary it speaks your framework: token ids may be a tensor of your
framework (or a list), and logits come back as a float32 tensor of your framework.

The API is functional, state in and state out, so it works unchanged from JAX:

    logits, cache = lm(tokens)                 # no cache: tokens are the whole prefix
    logits, cache = lm(next_tokens, cache)     # with cache: tokens continue the prefix

``logits`` has one row per token passed in *this* call: shape ``(T, vocab_size)`` for
1-D tokens of length T, or ``(B, T, vocab_size)`` for a batch ``(B, T)``.
``logits[..., -1, :]`` scores the next token. A ``KVCache`` is immutable: calling the
model never changes the cache you passed in, it returns a new one. You may therefore
reuse an old cache to branch (beam search, speculative decoding with rollback).

The model counts its work in ``lm.counters``:

    forward_calls      number of calls
    tokens_processed   total tokens passed in across calls (batch size times length)

Re-feeding the whole prefix at every step makes ``tokens_processed`` grow
quadratically; using the cache keeps it linear. That is what the cost judge measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters


@dataclass(frozen=True)
class KVCache:
    """Attention keys and values for every token seen so far. Opaque and immutable.

    ``keys[layer]`` and ``values[layer]`` have shape
    ``(batch, n_heads, length, head_dim)``.
    """

    keys: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    length: int
    batched: bool

    def __repr__(self) -> str:
        return f"KVCache(length={self.length}, layers={len(self.keys)})"


#: Standard deviation of the logits (the hidden state is normalized before the head).
LOGIT_STD = 2.0


def _rms_norm(x: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """RMSNorm: scale each vector to unit root-mean-square, then apply a learned gain."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return x / rms * gain


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class TinyLM:
    """A seeded decoder-only transformer. See the module docstring for the contract.

    Args:
        fw: the framework adapter the model speaks at its boundary (``ctx.fw``).
        seed: selects the weights; equal seeds give equal models.
        vocab_size, d_model, n_layers, n_heads, max_context: model size.

    Attributes:
        vocab_size, max_context: as given.
        bos_id: 0, the conventional first token of a prompt.
        eos_id: 1, the token that ends generation in problems that use one.
        counters: ``forward_calls`` and ``tokens_processed``.
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
        n_layers: int = 2,
        n_heads: int = 4,
        max_context: int = 128,
    ) -> None:
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.fw = fw
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_context = max_context
        self.counters = Counters("forward_calls", "tokens_processed")
        self._weights = self._init_weights()

    def _init_weights(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        d = self.d_model

        def normal(*shape: int, std: float) -> np.ndarray:
            return (rng.standard_normal(shape) * std).astype(np.float32)

        layers = []
        for _ in range(self.n_layers):
            layers.append(
                {
                    "norm1": np.ones(d, dtype=np.float32),
                    "wq": normal(d, d, std=d**-0.5),
                    "wk": normal(d, d, std=d**-0.5),
                    "wv": normal(d, d, std=d**-0.5),
                    "wo": normal(d, d, std=d**-0.5),
                    "norm2": np.ones(d, dtype=np.float32),
                    "w_in": normal(d, 4 * d, std=d**-0.5),
                    "w_out": normal(4 * d, d, std=(4 * d) ** -0.5),
                }
            )
        return {
            "tok": normal(self.vocab_size, d, std=1.0),
            "pos": normal(self.max_context, d, std=0.5),
            "layers": layers,
            "norm_out": np.ones(d, dtype=np.float32),
            # The output head. Its scale makes the next-token distribution clearly
            # non-uniform, so the argmax is separated by far more than float32 rounding.
            "head": normal(d, self.vocab_size, std=LOGIT_STD * d**-0.5),
        }

    def with_framework(self, fw: Any) -> TinyLM:
        """The same model (same weights) speaking another framework, with fresh counters."""
        twin = object.__new__(TinyLM)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("forward_calls", "tokens_processed")
        return twin

    # -- the forward pass --------------------------------------------------------

    def __call__(self, tokens: Any, cache: KVCache | None = None) -> tuple[Any, KVCache]:
        """Run the model on ``tokens``, continuing from ``cache`` if given.

        Returns ``(logits, new_cache)``. Raises ``ValueError`` for token ids outside
        the vocabulary, an empty call, or a total length beyond ``max_context``.
        """
        ids = self._token_array(tokens)
        batched = ids.ndim == 2
        if not batched:
            ids = ids[None, :]
        batch, new = ids.shape
        past = 0 if cache is None else cache.length
        if cache is not None and cache.keys[0].shape[0] != batch:
            raise ValueError(
                f"batch size {batch} does not match the cache's {cache.keys[0].shape[0]}"
            )
        if past + new > self.max_context:
            raise ValueError(f"sequence length {past + new} exceeds max_context {self.max_context}")
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", batch * new)

        w = self._weights
        heads, head_dim = self.n_heads, self.d_model // self.n_heads
        x = w["tok"][ids] + w["pos"][past : past + new]  # (B, T, D)

        # New token i (absolute position past + i) may attend to every cached
        # position and to new tokens up to and including itself.
        allowed = np.tril(np.ones((new, past + new), dtype=bool), k=past)

        new_keys, new_values = [], []
        for index, layer in enumerate(w["layers"]):
            h = _rms_norm(x, layer["norm1"])

            def split(t: np.ndarray) -> np.ndarray:  # (B, T, D) -> (B, H, T, Dh)
                return t.reshape(batch, new, heads, head_dim).transpose(0, 2, 1, 3)

            q, k, v = split(h @ layer["wq"]), split(h @ layer["wk"]), split(h @ layer["wv"])
            if cache is not None:
                k = np.concatenate([cache.keys[index], k], axis=2)
                v = np.concatenate([cache.values[index], v], axis=2)
            new_keys.append(k)
            new_values.append(v)

            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(np.float32(head_dim))
            scores = np.where(allowed, scores, np.float32(-1e9))
            attended = _softmax(scores) @ v  # (B, H, T, Dh)
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, new, self.d_model)
            x = x + attended @ layer["wo"]

            h = _rms_norm(x, layer["norm2"])
            x = x + _gelu(h @ layer["w_in"]) @ layer["w_out"]

        x = _rms_norm(x, w["norm_out"])
        logits = (x @ w["head"]).astype(np.float32)
        if not batched:
            logits = logits[0]

        for array in (*new_keys, *new_values):
            array.setflags(write=False)
        new_cache = KVCache(tuple(new_keys), tuple(new_values), past + new, batched)
        return self.fw.tensor(logits), new_cache

    def _token_array(self, tokens: Any) -> np.ndarray:
        if self.fw.is_tensor(tokens):
            tokens = self.fw.numpy(tokens)
        ids = np.asarray(tokens)
        if ids.ndim not in (1, 2) or ids.size == 0:
            raise ValueError(f"tokens must have shape (T,) or (B, T) with T >= 1, got {ids.shape}")
        if ids.dtype.kind not in "iu":
            raise ValueError(f"tokens must be integers, got dtype {ids.dtype}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        return ids.astype(np.int64)

    def __repr__(self) -> str:
        return (
            f"TinyLM(seed={self.seed}, vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}, max_context={self.max_context})"
        )
