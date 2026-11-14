# BUG: the weights are charged once per sequence (batch * (weight_bytes + kv_bytes_read) per step), so batching appears to gain nothing.
GB = 10**9


def decode_tokens_per_s(
    weight_bytes: int, kv_bytes_read: int, bandwidth_gb_per_s: float, batch: int
) -> float:
    bytes_per_step = batch * (weight_bytes + kv_bytes_read)
    steps_per_s = bandwidth_gb_per_s * GB / bytes_per_step
    return batch * steps_per_s


def crossover_batch(
    weight_bytes: int,
    kv_bytes_read: int,
    bandwidth_gb_per_s: float,
    compute_tokens_per_s: float,
) -> int | None:
    # With this model the rate does not depend on the batch: it crosses at 1 or never.
    rate = bandwidth_gb_per_s * GB / (weight_bytes + kv_bytes_read)
    return 1 if rate >= compute_tokens_per_s else None
