# BUG: goodput returns the highest rate that passes on its own, even when a lower rate failed: a lucky run above the knee is reported as capacity.
METRICS = ("ttft", "tpot", "e2e")


def is_good(request: dict, slo: dict) -> bool:
    """Did this request succeed and meet every objective in `slo`?"""
    if not request["ok"]:
        return False  # an error is never good, however quickly it came back
    for metric in METRICS:
        if metric not in slo:
            continue  # no objective on this metric
        value = request[metric]
        if value is None:
            continue  # tpot of a one-token response: there was no gap to be slow
        if value > slo[metric]:  # exactly at the objective still meets it
            return False
    return True


def slo_attainment(requests: list[dict], slo: dict) -> float:
    if not requests:
        raise ValueError("attainment of no requests is undefined")
    # Per request, then counted: an SLO is a statement about each request, not the mean.
    good = sum(is_good(request, slo) for request in requests)
    return good / len(requests)


def request_goodput(requests: list[dict], slo: dict, duration_s: float) -> float:
    good = sum(is_good(request, slo) for request in requests)
    return good / duration_s


def goodput(runs: dict[float, list[dict]], slo: dict, target: float) -> float | None:
    best = None
    for rate in sorted(runs):  # ascending, whatever order the runs were made in
        if slo_attainment(runs[rate], slo) >= target:
            best = rate
    return best
