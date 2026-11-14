"""MLXTinyLM: the kit's tiny language model written in ``mlx.nn``.

``TinyLM`` computes in numpy, which is right for judging but useless for problems
about MLX itself: laziness, ``mx.compile``, evaluation placement, batching. This is the
same kind of model (seeded weights, pre-norm blocks, a KV cache) built from MLX
operations, so its outputs are lazy ``mx.array`` graphs and it can be traced.

    lm = MLXTinyLM(seed=0)                       # vocab 64, 2 layers, 4 heads, RoPE
    logits, cache = lm(tokens)                   # tokens: (T,) or (B, T) integer array
    logits, cache = lm(next_tokens, cache)       # continue; logits: (T, V) or (B, T, V)

Nothing is computed until you evaluate. ``cache`` is a plain list with one
``(keys, values)`` pair per layer, each of shape ``(B, n_kv_heads, length, head_dim)``,
so ``mx.eval(logits, cache)`` evaluates all of it and it can cross an ``mx.compile``
boundary. The cache grows by concatenation: shapes change every step, which is fine
for an uncompiled loop and forces a retrace per step in a compiled one.

For compiled decode loops there is a second interface with fixed shapes:

    state = lm.init_state(max_len=64)            # preallocated buffers and a length
    logits, state = lm.step(token, state)        # token: (B,) ; logits: (B, V)

``state`` is a dict of arrays whose shapes never change, so a function built on
``step`` and compiled with ``mx.compile`` is traced once.

Two optional arguments of ``lm(...)`` serve batches of ragged prompts:
``mask`` (an additive mask broadcastable to ``(B, 1, T, length + T)``, replacing the
causal one) and ``positions`` (absolute positions for RoPE, ``(T,)`` or ``(B, T)``).

``lm.counters.forward_calls`` counts runs of the model's *Python body*, and
``tokens_processed`` the tokens those runs were given. In an ordinary loop that is
the number of calls. Inside ``mx.compile`` the body runs only when the function is
traced, so the same counter is the number of traces (see ``TraceCounter``).
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from whetstone.kit.counters import Counters

#: Standard deviation of the logits, as in TinyLM.
LOGIT_STD = 2.0


class _Block(nn.Module):
    def __init__(self, rng: np.random.Generator, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        head_dim = d_model // n_heads

        def linear(n_out: int, n_in: int, scale: float = 1.0) -> nn.Linear:
            layer = nn.Linear(n_in, n_out, bias=False)
            weight = rng.standard_normal((n_out, n_in)).astype(np.float32)
            layer.weight = mx.array(weight * np.float32(scale / math.sqrt(n_in)))
            return layer

        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.q_proj = linear(d_model, d_model, 2.0)
        self.k_proj = linear(n_kv_heads * head_dim, d_model, 2.0)
        self.v_proj = linear(n_kv_heads * head_dim, d_model)
        self.o_proj = linear(d_model, d_model)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.gate_proj = linear(2 * d_model, d_model)
        self.up_proj = linear(2 * d_model, d_model)
        self.down_proj = linear(d_model, 2 * d_model)


class MLXTinyLM(nn.Module):
    """A tiny decoder-only transformer in ``mlx.nn``. See the module docstring.

    Arguments: ``seed`` (the weights are a pure function of it and the sizes),
    ``vocab_size``, ``d_model``, ``n_layers``, ``n_heads``, ``n_kv_heads`` (default
    ``n_heads``; fewer gives grouped-query attention), ``rope`` (True: rotary
    positions on queries and keys; False: sinusoidal positions added to the
    embeddings), ``dtype`` (``mx.float32``; pass ``mx.float16`` for dtype problems).

    It is an ``nn.Module``: ``lm.parameters()``, ``lm.update(...)``, and
    ``nn.quantize(lm, ...)`` work as on any MLX model.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        rope: bool = True,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads or n_heads % n_kv_heads or (d_model // n_heads) % 2:
            raise ValueError(
                "d_model must be a multiple of n_heads, n_heads a multiple of n_kv_heads, "
                "and the head dimension even"
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.counters = Counters("forward_calls", "tokens_processed")

        rng = np.random.default_rng([int(seed), 0x4D4C58])
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight = mx.array(rng.standard_normal((vocab_size, d_model)).astype(np.float32))
        self.layers = [_Block(rng, d_model, n_heads, n_kv_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        head = rng.standard_normal((vocab_size, d_model)).astype(np.float32)
        self.head.weight = mx.array(head * np.float32(LOGIT_STD / math.sqrt(d_model)))
        if dtype is not None:
            self.set_dtype(dtype)
        mx.eval(self.parameters())

    # -- pieces ----------------------------------------------------------------------

    def _angles(self, positions: mx.array, width: int) -> mx.array:
        """``positions[..., None] * frequency`` for ``width / 2`` frequencies."""
        frequency = mx.exp(-mx.arange(0, width, 2).astype(mx.float32) * (math.log(10000.0) / width))
        return positions.astype(mx.float32)[..., None] * frequency

    def _rotate(self, x: mx.array, positions: mx.array) -> mx.array:
        """RoPE on ``(B, heads, T, head_dim)`` at ``positions`` of shape ``(B, T)``."""
        angles = self._angles(positions, self.head_dim)[:, None, :, :]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        half = self.head_dim // 2
        a, b = x[..., :half], x[..., half:]
        return mx.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    def _embed(self, tokens: mx.array, positions: mx.array) -> mx.array:
        x = self.embed(tokens)
        if not self.rope:
            angles = self._angles(positions, self.d_model)
            x = x + mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).astype(x.dtype)
        return x

    def _project(self, block: _Block, x: mx.array, positions: mx.array) -> tuple:
        """Queries, keys, values of the new tokens: ``(B, heads, T, head_dim)`` each."""
        batch, length, _ = x.shape
        h = block.norm1(x)

        def heads(layer: nn.Linear, n: int) -> mx.array:
            return layer(h).reshape(batch, length, n, self.head_dim).transpose(0, 2, 1, 3)

        q = heads(block.q_proj, self.n_heads)
        k = heads(block.k_proj, self.n_kv_heads)
        v = heads(block.v_proj, self.n_kv_heads)
        if self.rope:
            q, k = self._rotate(q, positions), self._rotate(k, positions)
        return q, k, v

    def _mix(self, block: _Block, x: mx.array, q: mx.array, k: mx.array, v: mx.array, mask: Any):
        """Attention over ``k``/``v`` (all tokens so far), then the MLP."""
        batch, length, _ = x.shape
        group = self.n_heads // self.n_kv_heads
        if group > 1:
            k, v = mx.repeat(k, group, axis=1), mx.repeat(v, group, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        mixed = mx.softmax(scores, axis=-1) @ v
        mixed = mixed.transpose(0, 2, 1, 3).reshape(batch, length, self.d_model)
        x = x + block.o_proj(mixed)
        h = block.norm2(x)
        return x + block.down_proj(nn.silu(block.gate_proj(h)) * block.up_proj(h))

    # -- the growing-cache interface -------------------------------------------------

    def __call__(
        self,
        tokens: Any,
        cache: list | None = None,
        *,
        mask: mx.array | None = None,
        positions: mx.array | None = None,
    ) -> tuple[mx.array, list]:
        tokens = tokens if isinstance(tokens, mx.array) else mx.array(tokens)
        if tokens.ndim not in (1, 2):
            raise ValueError("tokens must be a 1-D or 2-D integer array")
        batched = tokens.ndim == 2
        ids = tokens if batched else tokens[None, :]
        batch, length = ids.shape
        past = 0 if cache is None else cache[0][0].shape[2]
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", batch * length)

        if positions is None:
            positions = mx.arange(past, past + length)
        if positions.ndim == 1:
            positions = mx.broadcast_to(positions[None, :], (batch, length))
        if mask is None:
            rows = mx.arange(past, past + length)[:, None]
            cols = mx.arange(past + length)[None, :]
            mask = mx.where(cols <= rows, 0.0, -math.inf)[None, None, :, :]

        x = self._embed(ids, positions)
        new_cache = []
        for index, block in enumerate(self.layers):
            q, k, v = self._project(block, x, positions)
            if cache is not None:
                k = mx.concatenate([cache[index][0], k], axis=2)
                v = mx.concatenate([cache[index][1], v], axis=2)
            new_cache.append((k, v))
            x = self._mix(block, x, q, k, v, mask)
        logits = self.head(self.norm(x))
        return (logits if batched else logits[0]), new_cache

    # -- the fixed-shape interface (for mx.compile) ----------------------------------

    def init_state(self, max_len: int, batch_size: int = 1) -> dict[str, Any]:
        """An empty decode state with room for ``max_len`` tokens.

        ``{"keys": [...], "values": [...], "length": array}``: per layer a zero buffer
        of shape ``(batch_size, n_kv_heads, max_len, head_dim)``, and the number of
        tokens stored so far as an int32 scalar array. The shapes never change.
        """
        dtype = self.embed.weight.dtype
        shape = (batch_size, self.n_kv_heads, max_len, self.head_dim)
        return {
            "keys": [mx.zeros(shape, dtype=dtype) for _ in self.layers],
            "values": [mx.zeros(shape, dtype=dtype) for _ in self.layers],
            "length": mx.array(0, dtype=mx.int32),
        }

    def step(self, token: Any, state: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Feed one token per row: ``token`` of shape ``(B,)``, ``logits`` ``(B, V)``.

        Writes the new keys and values at index ``state["length"]`` and attends to
        everything up to it, with every shape fixed. Stepping past ``max_len`` is not
        detected (the write falls off the end and the token is lost).
        """
        token = token if isinstance(token, mx.array) else mx.array(token)
        ids = token.reshape(-1, 1)
        batch = ids.shape[0]
        length = state["length"]
        max_len = state["keys"][0].shape[2]
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", batch)

        positions = mx.broadcast_to(length.reshape(1, 1), (batch, 1))
        slots = mx.arange(max_len)
        here = (slots == length)[None, None, :, None]
        mask = mx.where(slots <= length, 0.0, -math.inf)[None, None, None, :]
        x = self._embed(ids, positions)
        keys, values = [], []
        for index, block in enumerate(self.layers):
            q, k, v = self._project(block, x, positions)
            k = mx.where(here, k, state["keys"][index])
            v = mx.where(here, v, state["values"][index])
            keys.append(k)
            values.append(v)
            x = self._mix(block, x, q, k, v, mask)
        logits = self.head(self.norm(x))[:, 0, :]
        return logits, {"keys": keys, "values": values, "length": length + 1}
