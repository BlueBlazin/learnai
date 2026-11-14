# BUG: an empty token list returns a dict of None values instead of raising ValueError.
def request_metrics(t_start: float, token_times: list[float]) -> dict:
    n = len(token_times)
    if n == 0:
        return {"ttft": None, "itl": [], "tpot": None, "e2e": None, "tokens_per_s": None}
    first, last = token_times[0], token_times[-1]
    e2e = last - t_start
    return {
        "ttft": first - t_start,
        # n tokens have n - 1 gaps between them; the wait for the first one is TTFT.
        "itl": [token_times[i] - token_times[i - 1] for i in range(1, n)],
        # The decode phase only: the first token's wait (queueing and prefill) is left
        # out of the time, and the first token is left out of the count.
        "tpot": (last - first) / (n - 1) if n > 1 else None,
        "e2e": e2e,
        # What the user experienced: every token, over the whole wait.
        "tokens_per_s": n / e2e if e2e > 0 else None,
    }
