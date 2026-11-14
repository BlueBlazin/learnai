def request_metrics(t_start: float, token_times: list[float]) -> dict:
    """The latency metrics of one streamed request.

    t_start: the time the request was sent, in seconds (any clock; only differences
        matter).
    token_times: the arrival time of each output token, in seconds on the same clock,
        in arrival order: n >= 1 values, non-decreasing, none before t_start. Must not
        be modified. An empty list raises ValueError.

    Returns a dict with exactly these keys:
        "ttft": float, time to first token: token_times[0] - t_start.
        "itl": list of n - 1 floats, the gaps between consecutive tokens
            (empty for n = 1).
        "tpot": float, time per output token with the first token excluded:
            (token_times[-1] - token_times[0]) / (n - 1); None for n = 1.
        "e2e": float, end-to-end latency: token_times[-1] - t_start.
        "tokens_per_s": float, output tokens per second over the whole request:
            n / e2e; None when e2e is 0.
    """
    raise NotImplementedError
