"""Numbers computed the same way in every problem that needs them."""

from __future__ import annotations

from typing import Any

import numpy as np


def logit_kl(p_logits: Any, q_logits: Any, axis: int = -1) -> np.ndarray:
    """KL divergence, in nats, between the distributions two sets of logits define.

        kl = logit_kl(original_logits, quantized_logits)      # shape (T,) for (T, V) inputs
        check.at_most("logit_kl", float(kl.mean()), 0.01, unit="nats")

    ``KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))`` with ``P = softmax(p_logits)``
    and ``Q = softmax(q_logits)`` along ``axis``: how much the predictions of a changed
    model (quantized, pruned, run in half precision) moved away from the original's,
    which is ``P``. Zero when the two are equal, and unchanged when a constant is added
    to either set of logits.

    Computed in float64 from log-softmax, so logits of any magnitude are safe. Inputs
    are numpy arrays (or anything ``np.asarray`` accepts) of the same shape; the result
    has that shape without ``axis``.
    """
    p = np.asarray(p_logits, dtype=np.float64)
    q = np.asarray(q_logits, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"logit_kl: shapes differ: {p.shape} and {q.shape}")
    log_p = _log_softmax(p, axis)
    log_q = _log_softmax(q, axis)
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=axis)


def _log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
