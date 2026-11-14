import math

GB = 10**9  # vendors quote bandwidth in decimal gigabytes per second


def decode_tokens_per_s(
    weight_bytes: int, kv_bytes_read: int, bandwidth_gb_per_s: float, batch: int
) -> float:
    # One step streams the weights once for the whole batch, plus every sequence's cache.
    bytes_per_step = weight_bytes + batch * kv_bytes_read
    steps_per_s = bandwidth_gb_per_s * GB / bytes_per_step
    return batch * steps_per_s  # each step emits one token per sequence


def crossover_batch(
    weight_bytes: int,
    kv_bytes_read: int,
    bandwidth_gb_per_s: float,
    compute_tokens_per_s: float,
) -> int | None:
    bandwidth = bandwidth_gb_per_s * GB
    # batch * bandwidth / (W + batch * K) >= C   <=>   batch * (bandwidth - C * K) >= C * W
    spare = bandwidth - compute_tokens_per_s * kv_bytes_read
    if spare <= 0:
        return None  # the bound saturates at bandwidth / K, below the compute rate
    return math.ceil(compute_tokens_per_s * weight_bytes / spare)
