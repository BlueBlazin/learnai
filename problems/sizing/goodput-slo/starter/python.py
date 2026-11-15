def slo_attainment(requests: list[dict], slo: dict) -> float:
    """The fraction of `requests` that are good under `slo`.

    requests: one dict per request with the keys "ok" (bool), "ttft", "tpot", and
        "e2e" (seconds; "tpot" is None for a one-token response, and all three may be
        None when "ok" is False).
    slo: objectives in seconds under any of the keys "ttft", "tpot", "e2e"; a metric
        that is not a key is unconstrained.
    Returns a float in [0, 1]. Raises ValueError if `requests` is empty.
    """
    raise NotImplementedError


def request_goodput(requests: list[dict], slo: dict, duration_s: float) -> float:
    """Good requests per second over a benchmark that lasted `duration_s` seconds.

    Returns a float; 0.0 for an empty list.
    """
    raise NotImplementedError


def goodput(runs: dict[float, list[dict]], slo: dict, target: float) -> float | None:
    """The highest sustainable request rate among the measured ones.

    runs: maps a request rate (requests per second) to the non-empty list of requests
        measured at that rate; the keys are in no particular order.
    target: the required attainment, in (0, 1].
    Returns the largest rate such that it and every lower measured rate have an
    attainment of at least `target`, or None if the lowest rate already misses it.
    """
    raise NotImplementedError
