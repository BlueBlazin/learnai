# BUG: the KV cache is left out of the bytes read per step, so the rate grows with the batch without limit and a crossover always exists.
import math

GB = 10**9


def decode_tokens_per_s(
    weight_bytes: int, kv_bytes_read: int, bandwidth_gb_per_s: float, batch: int
) -> float:
    steps_per_s = bandwidth_gb_per_s * GB / weight_bytes
    return batch * steps_per_s


def crossover_batch(
    weight_bytes: int,
    kv_bytes_read: int,
    bandwidth_gb_per_s: float,
    compute_tokens_per_s: float,
) -> int | None:
    bandwidth = bandwidth_gb_per_s * GB
    return math.ceil(compute_tokens_per_s * weight_bytes / bandwidth)
