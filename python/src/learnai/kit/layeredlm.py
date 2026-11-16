"""LayeredLM: a TinyLM whose layers can be run one range at a time.

``TinyLM`` is a black box: tokens in, logits out. Some inference techniques live
*inside* the stack. Early exit drafts tokens with the first few layers and verifies
them with the rest; KV sharing lets the upper layers read the keys and values of a
lower one, so that a prompt only has to run through the lower layers. ``LayeredLM`` is
the same model with the stack opened up:

    lm = LayeredLM(ctx.fw, seed=3)                    # 4 layers by default
    h = lm.embed(tokens, offset=cache.length)         # hidden states, (T, d_model)
    h, cache = lm.layers(h, 0, 2, cache)              # layers 0 and 1
    h, cache = lm.layers(h, 2, lm.n_layers, cache)    # layers 2 and 3
    logits = lm.head(lm.norm(h))                      # (T, vocab_size)

That sequence is exactly ``logits, cache = lm(tokens, cache)``, and for equal seeds and
sizes ``LayeredLM`` has the weights of ``TinyLM``. As there, the arithmetic is numpy,
tensors of the solution's framework go in and come out, and nothing is modified in
place: every call returns new hidden states and a new cache.

The cache
---------
A ``LayerCache`` holds, per layer, the attention keys and values of the tokens that
layer has processed. Layers may have processed different numbers of tokens (the early
layers of a self-speculating model run ahead of the late ones): ``cache.lengths`` is
the tuple of per-layer lengths, ``cache.length`` their maximum.
``cache.truncate(n)`` rolls every layer back to at most ``n`` tokens, and
``cache.select(rows)`` reorders batch rows, as on ``KVCache``.

``lm.layers(h, start, stop, cache)`` runs layers ``start`` to ``stop - 1``. The rows of
``h`` continue what those layers have cached, so the layers of one call must have equal
cached lengths (``ValueError`` otherwise), and ``h`` must be the hidden states of the
*next* positions, in order.

Sharing keys and values across layers
-------------------------------------
``lm.layers(h, start, stop, cache, kv_from=j)`` runs the layers as cross-readers of
layer ``j``: each computes its queries from its own input, but attends over the keys
and values layer ``j`` has cached, and stores none of its own. The ``T`` rows of ``h``
are the last ``T`` positions layer ``j`` has cached, so run layer ``j`` on them first.
``lm(tokens, cache, share_from=s)`` is the whole forward pass of the model in which
every layer from ``s`` up reads layer ``s - 1`` this way (the oracle for a problem
about that architecture).

Counters
--------
    layer_calls        layers run, summed over calls (a call of layers(h, 0, 2) adds 2)
    layer_tokens       layers run times tokens they were run on (batch size included)
    forward_calls, tokens_processed    whole forward passes through ``lm(...)``, as on
                                       TinyLM (those also add to the two above)

``embed``, ``norm``, and ``head`` are not counted: the cost of a layered technique is
which layers run on which tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from learnai.kit.counters import Counters
from learnai.kit.tinylm import TinyLM, _gelu, _rms_norm, _softmax

_COUNTERS = ("forward_calls", "tokens_processed", "layer_calls", "layer_tokens")


@dataclass(frozen=True)
class LayerCache:
    """Keys and values per layer; a layer that has cached nothing has ``None``.

    ``keys[i]`` and ``values[i]`` have shape ``(batch, n_heads, lengths[i], head_dim)``.
    Opaque and immutable, like ``KVCache``.
    """

    keys: tuple[np.ndarray | None, ...]
    values: tuple[np.ndarray | None, ...]
    batched: bool

    @property
    def lengths(self) -> tuple[int, ...]:
        """Tokens cached by each layer."""
        return tuple(0 if k is None else k.shape[2] for k in self.keys)

    @property
    def length(self) -> int:
        """The largest number of tokens any layer has cached."""
        return max(self.lengths, default=0)

    @property
    def batch_size(self) -> int | None:
        """Number of rows, or None for a cache that holds nothing yet."""
        for k in self.keys:
            if k is not None:
                return k.shape[0]
        return None

    def truncate(self, n: int) -> LayerCache:
        """Every layer rolled back to its first ``n`` tokens (layers with fewer keep
        what they have). ``0 <= n``, else ``ValueError``; a non-int is a ``TypeError``."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError(f"truncate expects an int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"cannot truncate a cache to {n} tokens")
        n = int(n)
        keys = tuple(None if k is None else k[:, :, :n] for k in self.keys)
        values = tuple(None if v is None else v[:, :, :n] for v in self.values)
        return LayerCache(keys, values, self.batched)

    def select(self, rows: Any) -> LayerCache:
        """Batch rows ``rows`` of this cache, in that order (see ``KVCache.select``)."""
        index = np.asarray(rows.tolist() if hasattr(rows, "tolist") else rows)
        if index.ndim != 1 or index.size == 0 or index.dtype.kind not in "iu":
            raise ValueError("rows must be a non-empty 1-D sequence of ints")
        size = self.batch_size
        if size is not None and (index.min() < 0 or index.max() >= size):
            raise ValueError(f"rows must be in [0, {size}), got {index.tolist()}")
        keys = tuple(None if k is None else _frozen(k[index]) for k in self.keys)
        values = tuple(None if v is None else _frozen(v[index]) for v in self.values)
        return LayerCache(keys, values, True)

    def __repr__(self) -> str:
        return f"LayerCache(lengths={self.lengths})"


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class LayeredLM(TinyLM):
    """A ``TinyLM`` with ``embed``, ``layers``, ``norm``, and ``head``. See the module docstring.

    Args:
        fw, seed, vocab_size, d_model, n_heads, max_context: as for ``TinyLM``.
        n_layers: 4 by default (``TinyLM`` has 2).
        residual_scale: a number, or one number per layer, multiplying what that
            layer adds to the hidden state (its attention and MLP outputs). With
            ``(1, 1, 0.25, 0.25)`` the late layers only refine what the early ones
            computed, so an early exit after layer 2 usually predicts the token the
            whole model predicts; with the default 1.0 the weights are ``TinyLM``'s
            and an early exit agrees with the full model only by chance.
    """

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        max_context: int = 128,
        residual_scale: float | Sequence[float] = 1.0,
    ) -> None:
        super().__init__(
            fw, seed, vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, max_context=max_context,
        )  # fmt: skip
        if isinstance(residual_scale, (int, float)):
            scales = (float(residual_scale),) * n_layers
        else:
            scales = tuple(float(s) for s in residual_scale)
        if len(scales) != n_layers:
            raise ValueError(f"residual_scale needs one number per layer ({n_layers})")
        self.residual_scale = scales
        self.counters = Counters(*_COUNTERS)

    def with_framework(self, fw: Any) -> LayeredLM:
        """The same model (same weights) speaking another framework, with fresh counters."""
        twin = object.__new__(type(self))
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters(*self.counters._names)
        return twin

    # -- the pieces ------------------------------------------------------------------------

    def embed(self, tokens: Any, offset: int = 0) -> Any:
        """Hidden states of ``tokens`` at positions ``offset, offset + 1, ...``.

        ``tokens`` is ``(T,)`` or ``(B, T)``; the result ``(T, d_model)`` or
        ``(B, T, d_model)``, float32. The model has learned position embeddings, so
        tokens that continue a cached prefix need ``offset=`` the number of tokens
        before them. Not counted.
        """
        ids = self._token_array(tokens)
        if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
            raise ValueError(f"offset must be an int >= 0, got {offset!r}")
        new = ids.shape[-1]
        if offset + new > self.max_context:
            raise ValueError(
                f"sequence length {offset + new} exceeds max_context {self.max_context}"
            )
        w = self._weights
        return self.fw.tensor(w["tok"][ids] + w["pos"][offset : offset + new])

    def layers(
        self,
        h: Any,
        start: int,
        stop: int,
        cache: LayerCache | None = None,
        *,
        kv_from: int | None = None,
    ) -> tuple[Any, LayerCache]:
        """Run layers ``start`` to ``stop - 1`` on hidden states ``h``.

        Returns ``(h_out, new_cache)``. Without ``kv_from`` each layer appends the keys
        and values of the new positions to its own cache. With ``kv_from=j`` the layers
        attend over layer ``j``'s cached keys and values and store nothing (module
        docstring). Raises ``ValueError`` for a range outside ``0 <= start < stop <=
        n_layers``, hidden states of the wrong shape, a batch size that does not match
        the cache, or layers whose cached lengths differ.
        """
        x, batched = self._hidden_array(h, "h")
        for name, value in (("start", start), ("stop", stop)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if not 0 <= start < stop <= self.n_layers:
            raise ValueError(f"layers({start}, {stop}): need 0 <= start < stop <= {self.n_layers}")
        if cache is None:
            cache = self.empty_cache(batched)
        if len(cache.keys) != self.n_layers:
            raise ValueError("this cache belongs to a model with another number of layers")
        batch, new, _ = x.shape
        if cache.batch_size not in (None, batch):
            raise ValueError(f"batch size {batch} does not match the cache's {cache.batch_size}")

        lengths = cache.lengths
        if kv_from is None:
            past = lengths[start]
            if any(lengths[i] != past for i in range(start, stop)):
                raise ValueError(
                    f"layers {start}..{stop - 1} have cached {lengths[start:stop]} tokens: "
                    "the layers of one call must continue from the same length"
                )
        else:
            if isinstance(kv_from, bool) or not isinstance(kv_from, (int, np.integer)):
                raise TypeError(f"kv_from must be a layer index, got {kv_from!r}")
            if not 0 <= kv_from < start:
                raise ValueError(f"kv_from={kv_from} must be a layer below start={start}")
            if lengths[kv_from] < new:
                raise ValueError(
                    f"layer {kv_from} has cached {lengths[kv_from]} tokens, fewer than the "
                    f"{new} rows of h: run layer {kv_from} on these positions first"
                )
            past = lengths[kv_from] - new
        if past + new > self.max_context:
            raise ValueError(f"sequence length {past + new} exceeds max_context {self.max_context}")

        count = int(stop - start)
        self.counters.add("layer_calls", count)
        self.counters.add("layer_tokens", count * batch * new)

        # New row i is absolute position past + i: it sees every earlier position and
        # the new ones up to itself.
        allowed = np.tril(np.ones((new, past + new), dtype=bool), k=past)
        keys, values = list(cache.keys), list(cache.values)
        for index in range(start, stop):
            if kv_from is None:
                x, keys[index], values[index] = self._block(
                    index, x, allowed, own=(keys[index], values[index])
                )
            else:
                x, _, _ = self._block(index, x, allowed, shared=(keys[kv_from], values[kv_from]))
        out = x if batched else x[0]
        return self.fw.tensor(out), LayerCache(tuple(keys), tuple(values), batched or cache.batched)

    def norm(self, h: Any) -> Any:
        """The final RMSNorm, which the head expects: ``logits = lm.head(lm.norm(h))``."""
        x, batched = self._hidden_array(h, "h")
        out = _rms_norm(x, self._weights["norm_out"]).astype(np.float32)
        return self.fw.tensor(out if batched else out[0])

    def head(self, h: Any) -> Any:
        """Logits ``(..., vocab_size)`` of *normalized* hidden states (see ``norm``).

        One output head serves every depth: the states after any layer can be
        normalized and projected, which is how an early exit predicts a token.
        """
        x, batched = self._hidden_array(h, "h")
        logits = (x @ self._weights["head"]).astype(np.float32)
        return self.fw.tensor(logits if batched else logits[0])

    def empty_cache(self, batched: bool = False) -> LayerCache:
        """A cache in which no layer has seen a token."""
        return LayerCache((None,) * self.n_layers, (None,) * self.n_layers, batched)

    # -- the whole forward pass ---------------------------------------------------------------

    def __call__(  # type: ignore[override]
        self, tokens: Any, cache: LayerCache | None = None, *, share_from: int | None = None
    ) -> tuple[Any, LayerCache]:
        """``tokens`` through every layer: ``(logits, new_cache)``, as on ``TinyLM``.

        With ``share_from=s`` (``1 <= s < n_layers``) the layers from ``s`` up read the
        keys and values of layer ``s - 1`` and the returned cache holds layers below
        ``s`` only.
        """
        ids = self._token_array(tokens)
        if share_from is not None and not (
            isinstance(share_from, (int, np.integer)) and 1 <= share_from < self.n_layers
        ):
            raise ValueError(f"share_from must be in [1, {self.n_layers}), got {share_from!r}")
        past = 0 if cache is None else cache.length
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", int(ids.size))
        numpy_twin = _plain(self)
        h = numpy_twin.embed(ids, offset=past)
        if share_from is None:
            h, cache = numpy_twin.layers(h, 0, self.n_layers, cache)
        else:
            h, cache = numpy_twin.layers(h, 0, share_from, cache)
            h, cache = numpy_twin.layers(
                h, share_from, self.n_layers, cache, kv_from=share_from - 1
            )
        return self.fw.tensor(numpy_twin.head(numpy_twin.norm(h))), cache

    # -- internals ------------------------------------------------------------------------------

    def _block(
        self,
        index: int,
        x: np.ndarray,
        allowed: np.ndarray,
        *,
        own: tuple[np.ndarray | None, np.ndarray | None] | None = None,
        shared: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """One pre-norm block: the arithmetic of ``TinyLM``, with the layer's scale."""
        layer = self._weights["layers"][index]
        scale = np.float32(self.residual_scale[index])
        batch, new, _ = x.shape
        heads, head_dim = self.n_heads, self.d_model // self.n_heads

        def split(t: np.ndarray) -> np.ndarray:  # (B, T, D) -> (B, H, T, Dh)
            return t.reshape(batch, new, heads, head_dim).transpose(0, 2, 1, 3)

        h = _rms_norm(x, layer["norm1"])
        q = split(h @ layer["wq"])
        if shared is not None:
            k, v = shared
        else:
            k, v = split(h @ layer["wk"]), split(h @ layer["wv"])
            old_k, old_v = own if own is not None else (None, None)
            if old_k is not None:
                k = np.concatenate([old_k, k], axis=2)
                v = np.concatenate([old_v, v], axis=2)
            _frozen(k), _frozen(v)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(np.float32(head_dim))
        scores = np.where(allowed, scores, np.float32(-1e9))
        attended = _softmax(scores) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, new, self.d_model)
        x = x + scale * (attended @ layer["wo"])
        x = x + scale * self._mlp(index, _rms_norm(x, layer["norm2"]))
        return x.astype(np.float32), k, v

    def _mlp(self, index: int, h: np.ndarray) -> np.ndarray:
        """The feed-forward half of layer ``index`` on normalized states ``(B, T, D)``."""
        layer = self._weights["layers"][index]
        return _gelu(h @ layer["w_in"]) @ layer["w_out"]

    def _hidden_array(self, h: Any, what: str) -> tuple[np.ndarray, bool]:
        """``h`` as a float32 array ``(B, T, D)``, and whether it came with a batch axis."""
        if self.fw.is_tensor(h):
            h = self.fw.numpy(h)
        x = np.asarray(h)
        if x.ndim not in (2, 3) or x.shape[-1] != self.d_model or x.shape[-2] == 0:
            raise ValueError(
                f"{what} must have shape (T, {self.d_model}) or (B, T, {self.d_model}) with "
                f"T >= 1, got {x.shape}"
            )
        if x.dtype.kind != "f":
            raise ValueError(f"{what} must be floating point, got dtype {x.dtype}")
        batched = x.ndim == 3
        return (x if batched else x[None]).astype(np.float32, copy=False), batched

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(seed={self.seed}, vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"max_context={self.max_context})"
        )


class _NumpyAdapter:
    """A do-nothing adapter: numpy arrays pass straight through."""

    name = "numpy"

    def is_tensor(self, value: Any) -> bool:
        return False

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        return value

    def numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)


def _plain(lm: LayeredLM) -> Any:
    """``lm`` speaking numpy and counting into ``lm``'s own counters."""
    twin = object.__new__(type(lm))
    twin.__dict__.update(lm.__dict__)
    twin.fw = _NumpyAdapter()
    return twin
