# BUG: converts the logits to NumPy to use boolean-mask indexing there and converts back; np.array(logits) evaluates the graph in the middle of the step.
import mlx.core as mx
import numpy as np


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    values = np.array(logits)
    if k >= values.shape[-1]:
        return logits
    kth = np.sort(values, axis=-1)[:, -k][:, None]
    values[values < kth] = -np.inf
    return mx.array(values)


def scatter_penalty(logits: mx.array, ids: mx.array, value: float) -> mx.array:
    values = np.array(logits)
    rows = np.arange(values.shape[0])[:, None]
    values[rows, np.array(ids)] -= value
    return mx.array(values)
