import mlx.core as mx

from learnai.kit import ops


def chain_logits(logits: mx.array, cfg: dict, history: mx.array) -> mx.array:
    # 1. Penalties, on the raw logits. counts[b, v] = how often token v occurs in row b
    #    of the history.
    vocab = mx.arange(logits.shape[-1])
    counts = (history[:, :, None] == vocab).sum(axis=1)
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    # Repetition penalty: always toward "less likely", so positive logits are divided
    # and the others multiplied.
    x = mx.where(seen, mx.where(logits > 0, logits / penalty, logits * penalty), logits)
    x = x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]

    # 2-4. The truncation filters. Each one sees the softmax of what the previous one
    #      left, because removed entries are -inf.
    x = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:  # 1.0 switches the stage off; do not let rounding trim the tail
        x = ops.top_p_filter(x, cfg["top_p"])
    x = ops.min_p_filter(x, cfg["min_p"])

    # 5. Temperature last: it reshapes the distribution over the survivors, but has no
    #    say in which tokens survive.
    return x / cfg["temperature"]


def sample_chain(
    logits: mx.array, cfg: dict, history: mx.array, u: mx.array
) -> mx.array:
    probs = ops.softmax(chain_logits(logits, cfg, history))
    # Inverse CDF: the first index whose cumulative sum exceeds u times the row total.
    cdf = mx.cumsum(probs, axis=-1)
    return (cdf <= u[:, None] * cdf[:, -1:]).sum(axis=-1)
