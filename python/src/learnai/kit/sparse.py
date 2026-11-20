"""CountingKV and TinyDSA: sparse attention whose savings can be measured.

DeepSeek Sparse Attention (DSA; DeepSeek-V3.2, GLM-5.3) does not attend to the whole
past. A cheap *indexer* scores every earlier token for every query, the ``k`` best are
selected, and attention runs over those ``k`` keys and values only. The saving is real
only if the other keys are never read: computing the full score matrix and masking it
gives the same numbers at the full cost. Two objects make that measurable.

``CountingKV`` is a key-value store that counts the rows read from it:

    store = CountingKV(ctx.fw, k, v)          # k (S, ...), v (S, ...)
    k_rows, v_rows = store.gather(idx, valid=idx >= 0)     # idx (T, kk): (T, kk, ...)
    store.counters.rows_read                  # how many rows that asked for
    store = store.append(k_new, v_new)        # a longer store; the counters carry over

``TinyDSA`` is a seeded stack of sparse-attention layers, handed out in pieces so that
a solution does the selecting, the gathering, and the caching itself:

    dsa = TinyDSA(ctx.fw, seed=1)             # 4 layers, d_model 32, types F S F S
    k_idx = dsa.index_key(layer, x)           # (T, index_dim); counts index_key_rows
    q_idx, w = dsa.index_query(layer, x)      # (T, index_heads, index_dim), (T, index_heads)
    q, k, v = dsa.qkv(layer, x)               # (T, n_heads, head_dim) each
    out = dsa.out(layer, attended)            # (T, n_heads, head_dim) -> (T, d_model)
    idx = dsa.indexer(layer, x)               # (T, top_k) token indices; counts indexer_calls
    out = dsa.attend(layer, x, idx)           # attention over the selected rows -> (T, d_model)

The arithmetic is numpy; the tensors at the boundary are the solution's.

What each layer computes for ``T`` tokens ``x`` of shape ``(T, d_model)``:

* indexer key, one per token: ``k_I = layer_norm(x Wk^T)`` (a LayerNorm with weight,
  bias, and eps 1e-6);
* indexer queries ``q_I = x Wq^T`` split into ``index_heads`` heads, and raw head
  weights ``w = x Ww^T``;
* index score ``I[t, s] = sum_h (w[t, h] * H^-0.5) * relu(q_I[t, h] . k_I[s] * D^-0.5)``
  with ``H = index_heads``, ``D = index_dim``; tokens after ``t`` are set to ``-inf``;
* selection: the ``min(k, S)`` highest scores of each row, best first, ties to the
  lowest index, and ``-1`` in a slot whose token is not visible (so the first rows of
  a sequence have trailing ``-1``);
* attention: ``softmax(q[t, h] . k[s, h] / sqrt(head_dim))`` over the selected ``s``
  only, times ``v``, then ``out``.

A token can always select itself. The stack oracle adds each layer's output to its
input, ``h = h + attend(i, h, idx)``, where ``idx`` is ``indexer(i, h)`` on a
``"full"`` layer and the nearest preceding full layer's indices on a ``"shared"``
layer (the cross-layer index reuse of GLM-5.3).

Compared with the shipped ``GlmMoeDsaIndexer`` (``transformers`` 5.23): scores and
selection are the same; left out are rotary positions on the indexer's first channels,
the low-rank query residual the indexer queries are projected from (here they come
from ``x``), and the latent (MLA) form of the main attention.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from learnai.kit.counters import Counters

NEG_INF = float("-inf")


def _array(fw: Any, value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(fw.numpy(value) if fw.is_tensor(value) else value, dtype=dtype)


class CountingKV:
    """Keys and values of ``n`` tokens; reading rows is counted.

    Args:
        fw: the framework adapter (``ctx.fw``).
        k, v: arrays or tensors whose first axis is the token: ``(n, ...)`` each, any
            trailing shapes, the same ``n``.

    Attributes:
        n: the number of rows.
        counters: ``rows_read`` (rows asked for over all ``gather`` calls) and
            ``gather_calls``. A store made by ``append`` shares the counters of the
            store it was made from, so one test reads the total of a whole decode.
        log: one ``(shape of rows, rows read)`` per ``gather`` call, in call order
            (shared like the counters).
    """

    def __init__(self, fw: Any, k: Any, v: Any) -> None:
        self.fw = fw
        self._k, self._v = _array(fw, k), _array(fw, v)
        if self._k.ndim < 1 or self._v.ndim < 1 or self._k.shape[0] != self._v.shape[0]:
            raise ValueError(
                f"k and v need the same number of rows, got {self._k.shape} and {self._v.shape}"
            )
        self.counters = Counters("rows_read", "gather_calls")
        self.log: list[tuple[tuple[int, ...], int]] = []

    @classmethod
    def empty(cls, fw: Any) -> CountingKV:
        """A store with no rows yet; the first ``append`` fixes the row shapes."""
        store = cls(fw, np.zeros((0,), np.float32), np.zeros((0,), np.float32))
        store._shapeless = True
        return store

    @property
    def n(self) -> int:
        return self._k.shape[0]

    def with_framework(self, fw: Any) -> CountingKV:
        """The same rows speaking another framework, with fresh counters and log."""
        twin = object.__new__(CountingKV)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("rows_read", "gather_calls")
        twin.log = []
        return twin

    def append(self, k: Any, v: Any) -> CountingKV:
        """A new store with the rows of ``k`` and ``v`` after this one's.

        ``k`` is ``(m, ...)`` with this store's key row shape, ``v`` likewise. This
        store is unchanged; the new one shares its counters and log. Appending reads
        nothing.
        """
        new_k, new_v = _array(self.fw, k), _array(self.fw, v)
        if new_k.ndim < 1 or new_v.ndim < 1 or new_k.shape[0] != new_v.shape[0]:
            raise ValueError(
                f"k and v need the same number of rows, got {new_k.shape} and {new_v.shape}"
            )
        longer = object.__new__(CountingKV)
        longer.__dict__.update(self.__dict__)
        if getattr(self, "_shapeless", False):
            longer._k, longer._v = new_k.copy(), new_v.copy()
            longer._shapeless = False
            return longer
        if new_k.shape[1:] != self._k.shape[1:] or new_v.shape[1:] != self._v.shape[1:]:
            raise ValueError(
                f"rows must have shapes {self._k.shape[1:]} (k) and {self._v.shape[1:]} (v), "
                f"got {new_k.shape[1:]} and {new_v.shape[1:]}"
            )
        longer._k = np.concatenate((self._k, new_k), axis=0)
        longer._v = np.concatenate((self._v, new_v), axis=0)
        return longer

    def gather(self, rows: Any, valid: Any = None) -> tuple[Any, Any]:
        """Read rows: ``(k_rows, v_rows)`` of shapes ``(*rows.shape, ...)``.

        ``rows`` is an integer array (or list) of any shape. Every entry is one row
        read and one count, also when it repeats a row. Indices follow Python:
        ``-1`` is the **last** row, and an index outside ``[-n, n)`` raises
        ``ValueError``.

        ``valid``, if given, is a boolean array of the same shape: where it is False
        the slot is not read and not counted, whatever its index, and comes back as
        zeros. That is how a selection padded with ``-1`` is read at the cost of its
        real entries: ``store.gather(idx, valid=idx >= 0)``.
        """
        index = _array(self.fw, rows, dtype=None)
        if index.dtype.kind not in "iu":
            raise ValueError(f"rows must be integers, got dtype {index.dtype}")
        index = index.astype(np.int64)
        if valid is None:
            keep = np.ones(index.shape, dtype=bool)
        else:
            keep = _array(self.fw, valid, dtype=None)
            if keep.dtype != np.bool_ or keep.shape != index.shape:
                raise ValueError(
                    f"valid must be a boolean array of shape {index.shape}, got "
                    f"{keep.dtype} {keep.shape}"
                )
        wanted = index[keep]
        if wanted.size and (wanted.min() < -self.n or wanted.max() >= self.n):
            raise ValueError(f"row indices must be in [-{self.n}, {self.n}), got {wanted.tolist()}")
        self.counters.add("gather_calls")
        self.counters.add("rows_read", int(wanted.size))
        self.log.append((tuple(index.shape), int(wanted.size)))
        k_rows = np.zeros((*index.shape, *self._k.shape[1:]), dtype=np.float32)
        v_rows = np.zeros((*index.shape, *self._v.shape[1:]), dtype=np.float32)
        if wanted.size:
            k_rows[keep], v_rows[keep] = self._k[wanted], self._v[wanted]
        return self.fw.tensor(k_rows), self.fw.tensor(v_rows)

    def reference(self) -> tuple[np.ndarray, np.ndarray]:
        """All keys and values as numpy copies, for tests. Not counted."""
        return self._k.copy(), self._v.copy()

    def __repr__(self) -> str:
        return f"CountingKV(n={self.n}, {self.counters!r})"


def index_types(n_layers: int, freq: int = 2, offset: int = 1) -> tuple[str, ...]:
    """GLM-5.3's default rule: layer ``i`` runs its own indexer (``"full"``) when
    ``max(i - offset + 1, 0) % max(freq, 1) == 0`` and is ``"shared"`` otherwise.

    GLM-5.3 itself has ``freq`` 4 and ``offset`` 3: three full layers, then three
    shared and one full, repeating.
    """
    return tuple(
        "full" if max(i - offset + 1, 0) % max(freq, 1) == 0 else "shared" for i in range(n_layers)
    )


class TinyDSA:
    """A seeded stack of sparse-attention layers. See the module docstring.

    Args:
        fw: the framework adapter (``ctx.fw``).
        seed: selects the weights.
        n_layers, d_model, n_heads, head_dim: the main attention (``n_heads * head_dim``
            need not equal ``d_model``).
        index_heads, index_dim: the indexer.
        top_k: the default selection size of ``indexer`` and the oracles.
        types: ``"full"`` or ``"shared"`` per layer; default ``index_types(n_layers)``,
            which is full, shared, full, shared, ...

    Attributes:
        counters: ``indexer_calls`` (calls of ``indexer``) and ``index_key_rows``
            (indexer keys computed: the rows passed to ``index_key``, plus every row
            of every ``indexer`` call, which computes its keys itself).
        types, top_k, n_layers, d_model, n_heads, head_dim, index_heads, index_dim.
    """

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        n_layers: int = 4,
        d_model: int = 32,
        n_heads: int = 2,
        head_dim: int = 8,
        index_heads: int = 4,
        index_dim: int = 8,
        top_k: int = 4,
        types: Any = None,
    ) -> None:
        for name, value in (("n_layers", n_layers), ("d_model", d_model), ("n_heads", n_heads),
                            ("head_dim", head_dim), ("index_heads", index_heads),
                            ("index_dim", index_dim), ("top_k", top_k)):  # fmt: skip
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        self.fw = fw
        self.seed = seed
        self.n_layers, self.d_model = n_layers, d_model
        self.n_heads, self.head_dim = n_heads, head_dim
        self.index_heads, self.index_dim = index_heads, index_dim
        self.top_k = top_k
        self.types = index_types(n_layers) if types is None else tuple(types)
        if len(self.types) != n_layers or set(self.types) - {"full", "shared"}:
            raise ValueError('types needs one of "full" / "shared" per layer')
        self.counters = Counters("indexer_calls", "index_key_rows")
        rng = np.random.default_rng([seed, 0x647361])
        width = n_heads * head_dim

        def linear(out_dim: int, in_dim: int, scale: float = 1.0) -> np.ndarray:
            return (rng.standard_normal((out_dim, in_dim)) * scale * in_dim**-0.5).astype(
                np.float32
            )

        self._layers = [
            {
                "index_q": linear(index_heads * index_dim, d_model),
                "index_k": linear(index_dim, d_model),
                "index_k_norm.weight": rng.uniform(0.5, 1.5, index_dim).astype(np.float32),
                "index_k_norm.bias": (rng.standard_normal(index_dim) * 0.1).astype(np.float32),
                "index_w": linear(index_heads, d_model),
                # Sharper than 1 / sqrt(d): which rows are selected then changes the output.
                "q": linear(width, d_model, 2.0),
                "k": linear(width, d_model),
                "v": linear(width, d_model),
                "o": linear(d_model, width, 0.5),
            }
            for _ in range(n_layers)
        ]
        for weights in self._layers:
            for array in weights.values():
                array.setflags(write=False)

    def with_framework(self, fw: Any) -> TinyDSA:
        """The same layers speaking another framework, with fresh counters."""
        twin = object.__new__(TinyDSA)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("indexer_calls", "index_key_rows")
        return twin

    # -- the pieces a solution composes ---------------------------------------------

    def index_key(self, layer: int, x: Any) -> Any:
        """The indexer keys of rows ``x``: ``(n, d_model)`` to ``(n, index_dim)``.

        One key per token, shared by all indexer heads. Counts ``n`` ``index_key_rows``:
        a decode loop should compute each token's key once and cache it.
        """
        index, rows = self._layer(layer), self._rows(x)  # a refused call counts nothing
        self.counters.add("index_key_rows", rows.shape[0])
        return self.fw.tensor(self._index_key(index, rows))

    def index_query(self, layer: int, x: Any) -> tuple[Any, Any]:
        """``(q_idx, w)`` for rows ``x``: ``(n, index_heads, index_dim)`` and ``(n, index_heads)``.

        ``w`` are the raw head weights; the score formula scales them by
        ``index_heads**-0.5``. Not counted.
        """
        q_idx, w = self._index_query(self._layer(layer), self._rows(x))
        return self.fw.tensor(q_idx), self.fw.tensor(w)

    def qkv(self, layer: int, x: Any) -> tuple[Any, Any, Any]:
        """Queries, keys, and values of rows ``x``: ``(n, n_heads, head_dim)`` each."""
        return tuple(self.fw.tensor(t) for t in self._qkv(self._layer(layer), self._rows(x)))

    def out(self, layer: int, attended: Any) -> Any:
        """The output projection: ``(n, n_heads, head_dim)`` to ``(n, d_model)``."""
        a = _array(self.fw, attended)
        if a.ndim != 3 or a.shape[1:] != (self.n_heads, self.head_dim):
            raise ValueError(
                f"attended must have shape (n, {self.n_heads}, {self.head_dim}), got {a.shape}"
            )
        return self.fw.tensor(self._out(self._layer(layer), a))

    def indexer(self, layer: int, x: Any, k: int | None = None) -> Any:
        """Run the indexer of ``layer`` on a whole sequence: ``(T, min(k, T))`` indices.

        Row ``t`` lists the ``k`` earlier-or-same tokens with the highest index scores,
        best first, ties to the lowest index, and ``-1`` in slots it cannot fill
        (``t + 1 < k``). ``k`` defaults to ``top_k``. An integer tensor of the
        framework. Counts one ``indexer_calls`` and ``T`` ``index_key_rows``.
        """
        index, rows, size = self._layer(layer), self._rows(x), self._k(k)
        self.counters.add("indexer_calls")
        self.counters.add("index_key_rows", rows.shape[0])
        return self.fw.tensor(self._select(index, rows, size))

    def attend(self, layer: int, x: Any, idx: Any) -> Any:
        """Attention of ``layer`` over the selected tokens only: ``(T, d_model)``.

        ``idx`` is ``(T, kk)`` token indices into the same ``T`` rows, ``-1`` for an
        unused slot: what ``indexer`` returns, for this layer or another. It is
        trusted: an index of a later token is attended to. Not counted.
        """
        index = self._layer(layer)
        rows = self._rows(x)
        chosen = _array(self.fw, idx, dtype=None)
        if chosen.ndim != 2 or chosen.shape[0] != rows.shape[0] or chosen.dtype.kind not in "iu":
            raise ValueError(
                f"idx must be integers of shape ({rows.shape[0]}, kk), got "
                f"{chosen.dtype} {chosen.shape}"
            )
        if chosen.size and (chosen.min() < -1 or chosen.max() >= rows.shape[0]):
            raise ValueError(f"idx entries must be -1 or in [0, {rows.shape[0]})")
        return self.fw.tensor(self._attend(index, rows, chosen.astype(np.int64)))

    # -- oracles: numpy in, numpy out, nothing counted ----------------------------------

    def index_scores_ref(self, layer: int, x: Any) -> np.ndarray:
        """The causal index scores of a sequence: ``(T, T)``, ``-inf`` above the diagonal.

        Use it to pick fixtures: a selection is only stable across frameworks when the
        ``k``-th and ``(k + 1)``-th scores of every row are clearly apart (or exactly
        equal, which the lowest index settles; exact zeros are common, because of the
        ReLU).
        """
        return self._scores(self._layer(layer), np.asarray(x, dtype=np.float32))

    def indexer_ref(self, layer: int, x: Any, k: int | None = None) -> np.ndarray:
        """What ``indexer`` returns, as int64 numpy."""
        return self._select(self._layer(layer), np.asarray(x, dtype=np.float32), self._k(k))

    def attend_ref(self, layer: int, x: Any, idx: Any) -> np.ndarray:
        """What ``attend`` returns, as float32 numpy."""
        return self._attend(
            self._layer(layer), np.asarray(x, dtype=np.float32), np.asarray(idx, dtype=np.int64)
        )

    def layer_ref(self, layer: int, x: Any, k: int | None = None) -> np.ndarray:
        """One layer on a whole sequence: ``attend(layer, x, indexer(layer, x, k))``.

        The full causal pass that a chunked decode with an indexer cache must
        reproduce, whatever the chunking.
        """
        rows = np.asarray(x, dtype=np.float32)
        return self._attend(self._layer(layer), rows, self.indexer_ref(layer, rows, k))

    def forward_ref(self, h: Any, types: Any = None, k: int | None = None) -> np.ndarray:
        """The stack: ``h = h + attend(i, h, idx)`` for every layer, sharing indices.

        ``types`` defaults to ``self.types``. A ``"full"`` layer selects with its own
        indexer; a ``"shared"`` layer uses the indices of the nearest full layer
        before it, and raises ``ValueError`` if there is none.
        """
        kinds = self.types if types is None else tuple(types)
        if len(kinds) != self.n_layers or set(kinds) - {"full", "shared"}:
            raise ValueError('types needs one of "full" / "shared" per layer')
        rows = np.asarray(h, dtype=np.float32)
        idx = None
        for i, kind in enumerate(kinds):
            if kind == "full":
                idx = self.indexer_ref(i, rows, k)
            elif idx is None:
                raise ValueError(f"layer {i} is shared, but no full layer comes before it")
            rows = rows + self._attend(i, rows, idx)
        return rows

    # -- arithmetic -------------------------------------------------------------------

    def _index_key(self, layer: int, x: np.ndarray) -> np.ndarray:
        weights = self._layers[layer]
        raw = x @ weights["index_k"].T
        centered = raw - raw.mean(axis=-1, keepdims=True)
        normed = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-6)
        key = normed * weights["index_k_norm.weight"] + weights["index_k_norm.bias"]
        return key.astype(np.float32)

    def _index_query(self, layer: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = self._layers[layer]
        q_idx = (x @ weights["index_q"].T).reshape(-1, self.index_heads, self.index_dim)
        return q_idx.astype(np.float32), (x @ weights["index_w"].T).astype(np.float32)

    def _scores(self, layer: int, x: np.ndarray) -> np.ndarray:
        q_idx, w = self._index_query(layer, x)
        k_idx = self._index_key(layer, x)
        per_head = np.maximum(np.einsum("thd,sd->ths", q_idx, k_idx) * self.index_dim**-0.5, 0.0)
        scores = np.einsum("th,ths->ts", w * self.index_heads**-0.5, per_head)
        steps = x.shape[0]
        visible = np.arange(steps)[None, :] <= np.arange(steps)[:, None]
        return np.where(visible, scores, NEG_INF).astype(np.float32)

    def _select(self, layer: int, x: np.ndarray, k: int) -> np.ndarray:
        scores = self._scores(layer, x)
        order = np.argsort(-scores, axis=-1, kind="stable")[:, : min(k, scores.shape[-1])]
        picked = np.take_along_axis(scores, order, axis=-1)
        return np.where(picked == NEG_INF, -1, order).astype(np.int64)

    def _qkv(self, layer: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._layers[layer]
        shape = (-1, self.n_heads, self.head_dim)
        return tuple((x @ weights[name].T).reshape(shape).astype(np.float32) for name in "qkv")

    def _out(self, layer: int, attended: np.ndarray) -> np.ndarray:
        flat = attended.reshape(attended.shape[0], self.n_heads * self.head_dim)
        return (flat @ self._layers[layer]["o"].T).astype(np.float32)

    def _attend(self, layer: int, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        q, k, v = self._qkv(layer, x)
        valid = idx >= 0
        rows = np.where(valid, idx, 0)
        scores = np.einsum("thd,tjhd->thj", q, k[rows]) / math.sqrt(self.head_dim)
        scores = np.where(valid[:, None, :], scores, NEG_INF)
        top = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - np.where(top == NEG_INF, 0.0, top))
        total = e.sum(axis=-1, keepdims=True)
        weights = e / np.where(total > 0, total, 1.0)
        return self._out(layer, np.einsum("thj,tjhd->thd", weights, v[rows]).astype(np.float32))

    # -- argument checks ----------------------------------------------------------------

    def _rows(self, x: Any) -> np.ndarray:
        rows = _array(self.fw, x, dtype=None)
        if rows.ndim != 2 or rows.shape[1] != self.d_model or rows.dtype.kind != "f":
            raise ValueError(
                f"x must be float rows of shape (n, {self.d_model}), got {rows.dtype} {rows.shape}"
            )
        return rows.astype(np.float32)

    def _layer(self, layer: Any) -> int:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            raise TypeError(f"layer must be an int, got {layer!r}")
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is outside [0, {self.n_layers})")
        return int(layer)

    def _k(self, k: Any) -> int:
        if k is None:
            return self.top_k
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        return int(k)

    def __repr__(self) -> str:
        return (
            f"TinyDSA(seed={self.seed}, n_layers={self.n_layers}, d_model={self.d_model}, "
            f"top_k={self.top_k}, types={self.types})"
        )
