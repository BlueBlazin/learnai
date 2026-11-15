# BUG: the objectives are checked against the mean of each metric over the successful requests, so attainment is all or nothing: one fast majority hides the slow requests, or one outlier fails everyone.
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
    done = [request for request in requests if request["ok"]]
    for metric in METRICS:
        values = [r[metric] for r in done if r[metric] is not None]
        if metric in slo and values and sum(values) / len(values) > slo[metric]:
            return 0.0
    return len(done) / len(requests)


def request_goodput(requests: list[dict], slo: dict, duration_s: float) -> float:
    good = sum(is_good(request, slo) for request in requests)
    return good / duration_s


def goodput(runs: dict[float, list[dict]], slo: dict, target: float) -> float | None:
    best = None
    for rate in sorted(runs):  # ascending, whatever order the runs were made in
        if slo_attainment(runs[rate], slo) < target:
            break  # everything above the first failing rate is out, even if it passes
        best = rate
    return best
