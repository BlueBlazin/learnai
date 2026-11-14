import math


def decode_tokens_per_s(
    weight_bytes: int, kv_bytes_read: int, bandwidth_gb_per_s: float, batch: int
) -> float:
    """Upper bound on decode throughput set by memory bandwidth.

    weight_bytes: bytes of weights, read once per decode step whatever the batch.
    kv_bytes_read: bytes of KV cache read per step for ONE sequence (>= 0).
    bandwidth_gb_per_s: memory bandwidth in GB/s, where 1 GB = 10**9 bytes.
    batch: number of sequences decoded together, >= 1.
    Returns a float: tokens per second summed over the whole batch (each step emits
    one token per sequence).
    """
    raise NotImplementedError


def crossover_batch(
    weight_bytes: int,
    kv_bytes_read: int,
    bandwidth_gb_per_s: float,
    compute_tokens_per_s: float,
) -> int | None:
    """Smallest batch at which decoding stops being memory-bound.

    compute_tokens_per_s: tokens per second, summed over the batch, that the
        processor's arithmetic could deliver if memory were free (> 0).
    Returns the smallest int batch >= 1 whose decode_tokens_per_s is at least
    compute_tokens_per_s, or None if no batch reaches it.
    """
    raise NotImplementedError
