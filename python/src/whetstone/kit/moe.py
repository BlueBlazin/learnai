"""CountingExperts and TinyMoE: mixture-of-experts layers whose expert work is counted.

A mixture-of-experts (MoE) layer replaces one MLP by many, of which every token uses
only a few, chosen by a router. The arithmetic can be written densely (run every
expert on every token and multiply by a weight that is mostly zero), which is correct
and wastes almost all of the work. An engine *dispatches*: it groups the tokens by
expert and runs each expert once, on exactly its tokens. ``CountingExperts`` makes the
difference measurable:

    experts = CountingExperts(ctx.fw, seed=1, n_experts=8, d_model=32)
    y = experts(3, rows)                  # expert 3 on rows (n, d_model) -> (n, d_model)
    experts.counters.expert_calls         # 1
    experts.counters.expert_rows          # n

Each expert is a SwiGLU MLP, ``(silu(x Wg^T) * (x Wu^T)) Wd^T``, with its own seeded
weights; a row's output depends on that row only. As everywhere in the kit, the
arithmetic is numpy and the tensors at the boundary are the solution's.

``moe_route_ref`` and ``moe_dense_ref`` are the oracles: the routing rule of the
``moe-router-forward`` problem (softmax over the router logits, the ``k`` largest
logits per token with ties to the lowest index, weights renormalized over the kept
experts) and the dense evaluation of the whole layer.

``TinyMoE`` is a ``LayeredLM`` whose MLPs are such layers. It is there to give the
dispatch problems inputs from a real forward pass: hidden states whose routing is as
uneven as a model's, not uniform noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from whetstone.kit.counters import Counters
from whetstone.kit.layeredlm import LayeredLM


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class CountingExperts:
    """``n_experts`` seeded SwiGLU experts, called one at a time and counted.

    Args:
        fw: the framework adapter (``ctx.fw``).
        seed: selects the weights.
        n_experts, d_model, d_hidden: sizes.

    Attributes:
        n_experts, d_model, d_hidden: as given.
        counters: ``expert_calls`` (calls) and ``expert_rows`` (rows over all calls).
        calls: the log, one ``(expert, n_rows)`` per call, in call order.
        weights: ``(w_gate, w_up, w_down)`` as numpy arrays of shape ``(n_experts,
            d_hidden, d_model)``, ``(n_experts, d_hidden, d_model)``, and ``(n_experts,
            d_model, d_hidden)``, for tests; reading them is not counted.
    """

    def __init__(
        self, fw: Any, seed: int = 0, *, n_experts: int = 8, d_model: int = 32, d_hidden: int = 48
    ) -> None:
        if n_experts < 1 or d_model < 1 or d_hidden < 1:
            raise ValueError("n_experts, d_model, and d_hidden must be at least 1")
        self.fw = fw
        self.seed = seed
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        rng = np.random.default_rng([seed, 0x4D6F45])

        def normal(*shape: int, std: float) -> np.ndarray:
            return (rng.standard_normal(shape) * std).astype(np.float32)

        self.weights = (
            normal(n_experts, d_hidden, d_model, std=d_model**-0.5),
            normal(n_experts, d_hidden, d_model, std=d_model**-0.5),
            normal(n_experts, d_model, d_hidden, std=d_hidden**-0.5),
        )
        for array in self.weights:
            array.setflags(write=False)
        self.counters = Counters("expert_calls", "expert_rows")
        self.calls: list[tuple[int, int]] = []

    def with_framework(self, fw: Any) -> CountingExperts:
        """The same experts speaking another framework, with fresh counters and log."""
        twin = object.__new__(CountingExperts)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("expert_calls", "expert_rows")
        twin.calls = []
        return twin

    def __call__(self, expert: int, x: Any) -> Any:
        """Run expert ``expert`` on the rows of ``x``: ``(n, d_model)`` in and out.

        One counted call, ``n`` counted rows. ``n`` may be 0 (the call still counts).
        Raises ``ValueError`` for an expert outside ``[0, n_experts)`` or rows of
        another shape, ``TypeError`` for an expert that is not an int.
        """
        index = self._expert(expert)
        rows = self.fw.numpy(x) if self.fw.is_tensor(x) else np.asarray(x)
        if rows.ndim != 2 or rows.shape[1] != self.d_model or rows.dtype.kind != "f":
            raise ValueError(
                f"an expert takes float rows of shape (n, {self.d_model}), got "
                f"{rows.dtype} {rows.shape}"
            )
        self.counters.add("expert_calls")
        self.counters.add("expert_rows", rows.shape[0])
        self.calls.append((index, rows.shape[0]))
        return self.fw.tensor(self.reference(index, rows))

    def reference(self, expert: int, x: Any) -> np.ndarray:
        """Expert ``expert`` on numpy rows ``(..., d_model)``; the oracle, not counted."""
        index = self._expert(expert)
        w_gate, w_up, w_down = (w[index] for w in self.weights)
        rows = np.asarray(x, dtype=np.float32)
        return ((_silu(rows @ w_gate.T) * (rows @ w_up.T)) @ w_down.T).astype(np.float32)

    def _expert(self, expert: Any) -> int:
        if isinstance(expert, bool) or not isinstance(expert, (int, np.integer)):
            raise TypeError(f"the expert must be an int, got {expert!r}")
        if not 0 <= expert < self.n_experts:
            raise ValueError(f"expert {expert} is outside [0, {self.n_experts})")
        return int(expert)

    def __repr__(self) -> str:
        return f"CountingExperts(n_experts={self.n_experts}, {self.counters!r})"


def moe_route_ref(x: Any, w_router: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Which experts each token uses, and with what weights.

        experts, weights = moe_route_ref(x, w_router, k=2)      # both (n_tokens, k)

    ``x`` is ``(n_tokens, d_model)`` and ``w_router`` ``(n_experts, d_model)``, numpy.
    Router logits are ``x @ w_router.T``; a token keeps its ``k`` largest logits (ties
    to the lowest expert index), listed from the largest down, and its weights are the
    softmax probabilities of the kept experts divided by their sum. That is the rule
    of ``moe-router-forward``. Indices are int64, weights float32.
    """
    rows = np.asarray(x, dtype=np.float32)
    router = np.asarray(w_router, dtype=np.float32)
    if rows.ndim != 2 or router.ndim != 2 or rows.shape[1] != router.shape[1]:
        raise ValueError(f"x {rows.shape} and w_router {router.shape} do not fit together")
    if not 1 <= k <= router.shape[0]:
        raise ValueError(f"k must be in [1, {router.shape[0]}], got {k}")
    logits = (rows @ router.T).astype(np.float32)
    order = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    kept = np.take_along_axis(logits, order, axis=-1).astype(np.float64)
    kept = np.exp(kept - kept.max(axis=-1, keepdims=True))
    return order.astype(np.int64), (kept / kept.sum(axis=-1, keepdims=True)).astype(np.float32)


def moe_dense_ref(x: Any, w_router: Any, experts: CountingExperts, k: int) -> np.ndarray:
    """The MoE layer evaluated densely: the oracle a dispatching solution must match.

    ``sum_j weights[t, j] * expert_{experts[t, j]}(x[t])`` for every token ``t``, with
    the routing of ``moe_route_ref``. float32 ``(n_tokens, d_model)``; ``experts``'
    counters are not touched.
    """
    rows = np.asarray(x, dtype=np.float32)
    chosen, weights = moe_route_ref(rows, w_router, k)
    out = np.zeros_like(rows)
    for expert in range(experts.n_experts):
        share = np.where(chosen == expert, weights, 0.0).sum(axis=-1, dtype=np.float32)
        if share.any():
            out += share[:, None] * experts.reference(expert, rows)
    return out


class TinyMoE(LayeredLM):
    """A ``LayeredLM`` whose feed-forward layers are mixtures of experts.

        moe = TinyMoE(ctx.fw, seed=2, n_experts=8, top_k=2)
        x = moe.mlp_inputs(tokens, layer=1)        # (n_tokens, d_model): what the MoE of
        experts = moe.experts(1)                   #   layer 1 receives for these tokens
        w_router = moe.router(1)                   # (n_experts, d_model)
        y = ctx.call("moe_sparse", x, w_router, experts, moe.top_k)
        check.allclose(y, moe_dense_ref(x, w_router, experts, moe.top_k), ...)

    Each layer has its own router and its own ``CountingExperts``; the forward pass
    (``moe(tokens, cache)``, ``layers``, ...) evaluates them with the dense oracle and
    does not touch the experts' counters. Sizes as for ``LayeredLM`` (2 layers by
    default), plus ``n_experts``, ``top_k``, and the experts' ``d_hidden``.
    """

    def __init__(
        self,
        fw: Any,
        seed: int = 0,
        *,
        n_experts: int = 8,
        top_k: int = 2,
        d_hidden: int = 48,
        n_layers: int = 2,
        **model_args: Any,
    ) -> None:
        super().__init__(fw, seed, n_layers=n_layers, **model_args)
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, {n_experts}], got {top_k}")
        self.n_experts = n_experts
        self.top_k = top_k
        rng = np.random.default_rng([seed, 0x726F7574])
        self._routers = [
            (rng.standard_normal((n_experts, self.d_model)) * self.d_model**-0.5).astype(np.float32)
            for _ in range(n_layers)
        ]
        self._experts = [
            CountingExperts(
                fw, seed * 1000 + layer, n_experts=n_experts, d_model=self.d_model,
                d_hidden=d_hidden,
            )
            for layer in range(n_layers)
        ]  # fmt: skip

    def with_framework(self, fw: Any) -> TinyMoE:
        twin = super().with_framework(fw)
        twin._experts = [experts.with_framework(fw) for experts in self._experts]
        return twin  # type: ignore[return-value]

    def router(self, layer: int) -> Any:
        """The router weights of ``layer``: ``(n_experts, d_model)``, a tensor of the framework."""
        return self.fw.tensor(self._routers[self._layer(layer)])

    def experts(self, layer: int) -> CountingExperts:
        """The (counting) experts of ``layer``. The same object on every call."""
        return self._experts[self._layer(layer)]

    def mlp_inputs(self, tokens: Any, layer: int) -> np.ndarray:
        """The rows the MoE of ``layer`` receives when the model reads ``tokens``.

        A numpy array ``(n_tokens, d_model)`` (batch rows flattened): test input for a
        dispatch problem. Runs a forward pass that is not counted anywhere.
        """
        target = self._layer(layer)
        seen: list[np.ndarray] = []
        probe = self.with_framework(self.fw)
        probe._tap = (target, seen)
        probe(tokens)
        return seen[0].reshape(-1, self.d_model).copy()

    def _mlp(self, index: int, h: np.ndarray) -> np.ndarray:
        tap = getattr(self, "_tap", None)
        if tap is not None and tap[0] == index:
            tap[1].append(h.astype(np.float32))
        flat = h.reshape(-1, self.d_model)
        out = moe_dense_ref(flat, self._routers[index], self._experts[index], self.top_k)
        return out.reshape(h.shape)

    def _layer(self, layer: Any) -> int:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            raise TypeError(f"layer must be an int, got {layer!r}")
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is outside [0, {self.n_layers})")
        return int(layer)
