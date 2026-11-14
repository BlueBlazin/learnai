# BUG: TPOT is the end-to-end latency divided by all n tokens, so the wait for the first token (queueing and prefill) is averaged into the per-token decode time.
def request_metrics(t_start: float, token_times: list[float]) -> dict:
    n = len(token_times)
    if n == 0:
        raise ValueError("a request without output tokens has no latency metrics")
    first, last = token_times[0], token_times[-1]
    e2e = last - t_start
    return {
        "ttft": first - t_start,
        # n tokens have n - 1 gaps between them; the wait for the first one is TTFT.
        "itl": [token_times[i] - token_times[i - 1] for i in range(1, n)],
        # The decode phase only: the first token's wait (queueing and prefill) is left
        # out of the time, and the first token is left out of the count.
        "tpot": e2e / n if n > 1 else None,
        "e2e": e2e,
        # What the user experienced: every token, over the whole wait.
        "tokens_per_s": n / e2e if e2e > 0 else None,
    }
