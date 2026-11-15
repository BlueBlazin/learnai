def plan(cost: dict, workload: dict, tpot_slo_us: int, headroom_percent: int) -> dict | None:
    """Size a deployment from the engine's cost model.

    cost: int microseconds under "base_us", "prefill_us_per_token", and
        "decode_us_per_seq", and the int "max_seqs" (sequences a replica runs at once).
    workload: ints under "requests_per_s", "prompt_tokens", and "output_tokens" (the
        last two are the lengths every request is assumed to have).
    tpot_slo_us: the objective on the mean time per output token, in microseconds.
    headroom_percent: the share of a replica's capacity left unused, an int in 0..90.
    Returns None if even a batch of one misses the objective; otherwise a dict with
    "batch" and "replicas" (ints) and "tpot_us", "requests_per_s", "tokens_per_s",
    and "utilization" (floats).
    """
    raise NotImplementedError


def plan_from_curve(
    curve: list[tuple[int, int]], requests_per_s: int, slo_ms: int, headroom_percent: int
) -> dict | None:
    """Size a deployment from a measured single-replica latency curve.

    curve: (requests per second offered to one replica, measured latency in ms)
        pairs, by strictly increasing rate; all ints.
    Returns None if the first point misses the objective; otherwise a dict with
    "capacity" (float, requests per second per replica) and "replicas" (int).
    """
    raise NotImplementedError
