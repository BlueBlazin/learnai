# BUG: decode_tokens_per_s returns the steps per second, which is the rate of one sequence, instead of the total over the batch.
import math

GB = 10**9


def decode_tokens_per_s(
    weight_bytes: int, kv_bytes_read: int, bandwidth_gb_per_s: float, batch: int
) -> float:
    bytes_per_step = weight_bytes + batch * kv_bytes_read
    steps_per_s = bandwidth_gb_per_s * GB / bytes_per_step
    return steps_per_s


def crossover_batch(
    weight_bytes: int,
    kv_bytes_read: int,
    bandwidth_gb_per_s: float,
    compute_tokens_per_s: float,
) -> int | None:
    bandwidth = bandwidth_gb_per_s * GB
    spare = bandwidth - compute_tokens_per_s * kv_bytes_read
    if spare <= 0:
        return None
    return math.ceil(compute_tokens_per_s * weight_bytes / spare)
