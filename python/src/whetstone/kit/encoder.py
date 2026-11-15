"""TinyEncoder and CountingScorer: the models behind embedding and reranking problems.

Retrieval uses two kinds of model that a decoder-only ``TinyLM`` cannot stand in for.
An *embedding model* reads a text and returns a state per token, with attention in
both directions; pooling those states gives the text's vector. A *reranker* (a
cross-encoder) reads a query and a document together and returns one relevance score.
Both are small here, seeded, and numpy inside, like the rest of the kit.

    enc = TinyEncoder(ctx.fw, seed=1)
    states = enc(tokens, mask)             # (B, T, d_model) float32; mask: 1 = real token

    scorer = CountingScorer(seed=1, max_length=48)
    scores = scorer([(query, doc_a), (query, doc_b)])       # one float per pair
    scorer.counters.padded_tokens                            # 2 * (longest pair's length)

Padding and the encoder
-----------------------
A batch of texts of different lengths is padded, and ``mask`` says which positions are
real (1 or True) and which are padding. The encoder honors it: a real token attends to
real tokens only, and its position is counted among the real tokens of its row, so
padding on either side leaves the states of the real tokens unchanged (to float32
rounding, about ``1e-6``). The states *at* padded positions are whatever the
arithmetic gives for a padding token; a pooling function must not read them, which is
the point of the problem that uses this.

The scorer's batches
--------------------
A cross-encoder processes a batch as one rectangle: every pair is padded to the
longest pair of its batch. ``CountingScorer`` charges for the rectangle,

    padded_tokens += len(batch) * max(len(query) + len(doc) for the pairs of the batch)

so batching pairs of similar length together is cheaper than batching them as they
come. A pair's score depends on that pair alone, exactly (equal pairs get equal
floats, whatever batch they are in). A pair longer than ``max_length`` is refused:
truncating is the caller's decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters
from whetstone.kit.tinylm import _gelu, _rms_norm, _softmax


class TinyEncoder:
    """A seeded bidirectional transformer encoder. See the module docstring.

    Args:
        fw: the framework adapter (``ctx.fw``).
        seed: selects the weights.
        vocab_size, d_model, n_layers, n_heads, max_length: sizes.

    Attributes:
        pad_id: 0, the conventional padding token (the mask, not the id, is what the
            encoder goes by).
        cls_id: 1, the conventional first token of a text.
        counters: ``forward_calls``, ``tokens_processed`` (batch size times padded
            length), and ``real_tokens`` (positions where the mask is set).
    """

    pad_id = 0
    cls_id = 1

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_length: int = 64,
    ) -> None:
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.fw = fw
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_length = max_length
        self.counters = Counters("forward_calls", "tokens_processed", "real_tokens")
        rng = np.random.default_rng([seed, 0x656E63])
        d = d_model

        def normal(*shape: int, std: float) -> np.ndarray:
            return (rng.standard_normal(shape) * std).astype(np.float32)

        self._weights = {
            "tok": normal(vocab_size, d, std=1.0),
            "pos": normal(max_length, d, std=0.5),
            "layers": [
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
                for _ in range(n_layers)
            ],
            "norm_out": np.ones(d, dtype=np.float32),
        }

    def with_framework(self, fw: Any) -> TinyEncoder:
        """The same encoder speaking another framework, with fresh counters."""
        twin = object.__new__(TinyEncoder)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("forward_calls", "tokens_processed", "real_tokens")
        return twin

    def __call__(self, tokens: Any, mask: Any = None) -> Any:
        """Token states for ``tokens`` ``(T,)`` or ``(B, T)``: ``(T, d_model)`` or
        ``(B, T, d_model)``, float32.

        ``mask`` has the shape of ``tokens``; nonzero marks a real token. Without it
        every position is real. Raises ``ValueError`` for ids outside the vocabulary,
        a mask of another shape, a row without any real token, or more than
        ``max_length`` positions.
        """
        ids, real, batched = self._inputs(tokens, mask)
        batch, length = ids.shape
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", batch * length)
        self.counters.add("real_tokens", int(real.sum()))
        states = self._states(ids, real)
        return self.fw.tensor(states if batched else states[0])

    def reference(self, tokens: Any, mask: Any = None) -> np.ndarray:
        """The same states as a numpy array, without counting (for tests)."""
        ids, real, batched = self._inputs(tokens, mask)
        states = self._states(ids, real)
        return states if batched else states[0]

    def _inputs(self, tokens: Any, mask: Any) -> tuple[np.ndarray, np.ndarray, bool]:
        ids = np.asarray(self.fw.numpy(tokens) if self.fw.is_tensor(tokens) else tokens)
        if ids.ndim not in (1, 2) or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError(f"tokens must be integers of shape (T,) or (B, T), got {ids.shape}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        batched = ids.ndim == 2
        ids = ids if batched else ids[None]
        if ids.shape[1] > self.max_length:
            raise ValueError(f"{ids.shape[1]} positions exceed max_length {self.max_length}")
        if mask is None:
            real = np.ones(ids.shape, dtype=bool)
        else:
            given = np.asarray(self.fw.numpy(mask) if self.fw.is_tensor(mask) else mask)
            given = given if batched else given[None] if given.ndim == 1 else given
            if given.shape != ids.shape:
                raise ValueError(
                    f"mask {given.shape} does not have the shape of tokens {ids.shape}"
                )
            real = given != 0
        if not real.any(axis=1).all():
            raise ValueError("every row needs at least one real (unmasked) token")
        return ids.astype(np.int64), real, batched

    def _states(self, ids: np.ndarray, real: np.ndarray) -> np.ndarray:
        w = self._weights
        batch, length = ids.shape
        heads, head_dim = self.n_heads, self.d_model // self.n_heads
        # A token's position is its index among the real tokens of its row, so the
        # same text gets the same positions wherever its padding is.
        positions = np.clip(np.cumsum(real, axis=1) - 1, 0, None)
        x = w["tok"][ids] + w["pos"][positions]
        visible = real[:, None, None, :]  # every query may attend to the real keys

        def split(t: np.ndarray) -> np.ndarray:
            return t.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        for layer in w["layers"]:
            h = _rms_norm(x, layer["norm1"])
            q, k, v = split(h @ layer["wq"]), split(h @ layer["wk"]), split(h @ layer["wv"])
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(np.float32(head_dim))
            scores = np.where(visible, scores, np.float32(-1e9))
            attended = (_softmax(scores) @ v).transpose(0, 2, 1, 3)
            x = x + attended.reshape(batch, length, self.d_model) @ layer["wo"]
            x = x + _gelu(_rms_norm(x, layer["norm2"]) @ layer["w_in"]) @ layer["w_out"]
        return _rms_norm(x, w["norm_out"]).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"TinyEncoder(seed={self.seed}, vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, n_layers={self.n_layers}, max_length={self.max_length})"
        )


class _Numpy:
    name = "numpy"

    def is_tensor(self, value: Any) -> bool:
        return False

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        return value

    def numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)


Pair = tuple[Sequence[int], Sequence[int]]


class CountingScorer:
    """A seeded cross-encoder: ``scorer(pairs)`` returns one relevance score per pair.

    Args:
        seed: selects the weights.
        max_length: the longest pair (query tokens plus document tokens) it accepts.
        max_batch: the largest batch it accepts; None for no limit.
        vocab_size: token ids are in ``[0, vocab_size)``.

    A pair is ``(query_tokens, doc_tokens)``, two non-empty sequences of ints. Scores
    are Python floats, a deterministic function of the pair alone. Counters:
    ``calls``, ``pairs``, ``padded_tokens`` (the batch rectangle: pairs times the
    longest pair), and ``real_tokens`` (what the pairs actually hold). ``batches`` is
    the log: the pair lengths of every call. ``scorer.score(query, doc)`` is the
    oracle for one pair, not counted.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        max_length: int = 64,
        max_batch: int | None = None,
        vocab_size: int = 64,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2: a query token and a document token")
        self.seed = seed
        self.max_length = max_length
        self.max_batch = max_batch
        self.vocab_size = vocab_size
        self.counters = Counters("calls", "pairs", "padded_tokens", "real_tokens")
        self.batches: list[list[int]] = []
        # Two segments, told apart by a learned segment vector, read by one encoder.
        self._encoder = TinyEncoder(
            _Numpy(), seed, vocab_size=vocab_size, max_length=max_length, n_layers=1
        )
        rng = np.random.default_rng([seed, 0x72616E6B])
        d = self._encoder.d_model
        self._segment = (rng.standard_normal((2, d)) * 0.5).astype(np.float32)
        self._readout = (rng.standard_normal(d) * d**-0.5).astype(np.float32)
        self._scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    def with_framework(self, fw: Any) -> CountingScorer:
        """The same scorer with fresh counters."""
        twin = object.__new__(CountingScorer)
        twin.__dict__.update(self.__dict__)
        twin.counters = Counters("calls", "pairs", "padded_tokens", "real_tokens")
        twin.batches = []
        return twin

    def __call__(self, pairs: Sequence[Pair]) -> list[float]:
        """Score a batch of pairs. Raises ``ValueError`` for an empty batch, a batch
        over ``max_batch``, a pair over ``max_length``, or an empty query or document."""
        batch = [self._pair(pair) for pair in pairs]
        if not batch:
            raise ValueError("the scorer was called with an empty batch")
        if self.max_batch is not None and len(batch) > self.max_batch:
            raise ValueError(f"a batch of {len(batch)} pairs exceeds max_batch {self.max_batch}")
        lengths = [len(query) + len(doc) for query, doc in batch]
        self.counters.add("calls")
        self.counters.add("pairs", len(batch))
        self.counters.add("padded_tokens", len(batch) * max(lengths))
        self.counters.add("real_tokens", sum(lengths))
        self.batches.append(lengths)
        return [self._score(query, doc) for query, doc in batch]

    def score(self, query: Sequence[int], doc: Sequence[int]) -> float:
        """The score of one pair; the oracle, not counted."""
        return self._score(*self._pair((query, doc)))

    def _pair(self, pair: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        try:
            query, doc = pair
            query, doc = tuple(int(t) for t in query), tuple(int(t) for t in doc)
        except TypeError, ValueError:
            raise ValueError(
                "a pair is (query_tokens, doc_tokens), two sequences of ints"
            ) from None
        if not query or not doc:
            raise ValueError("a pair needs at least one query token and one document token")
        if len(query) + len(doc) > self.max_length:
            raise ValueError(
                f"a pair of {len(query)} + {len(doc)} tokens exceeds max_length "
                f"{self.max_length}: truncate the document first"
            )
        if min(query + doc) < 0 or max(query + doc) >= self.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        return query, doc

    def _score(self, query: tuple[int, ...], doc: tuple[int, ...]) -> float:
        key = (query, doc)
        if key not in self._scores:
            # Each pair is encoded by itself, unpadded: its score cannot depend on
            # what it was batched with.
            ids = np.asarray(query + doc, dtype=np.int64)[None]
            segments = np.repeat(np.arange(2), (len(query), len(doc)))
            states = self._encode(ids, np.ones(ids.shape, dtype=bool), segments)
            self._scores[key] = float(np.tanh(states[0].mean(axis=0) @ self._readout))
        return self._scores[key]

    def _encode(self, ids: np.ndarray, real: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """The encoder's states with the segment vectors added to the embeddings."""
        encoder = self._encoder
        w = encoder._weights
        shifted = dict(w)
        # One extra embedding row per (token, segment) would be the textbook layout;
        # adding the segment vector to the position table of this one call is equal.
        length = ids.shape[1]
        shifted["pos"] = w["pos"].copy()
        shifted["pos"][:length] += self._segment[segments]
        encoder._weights = shifted
        try:
            return encoder._states(ids, real)
        finally:
            encoder._weights = w

    def __repr__(self) -> str:
        return f"CountingScorer(max_length={self.max_length}, {self.counters!r})"
