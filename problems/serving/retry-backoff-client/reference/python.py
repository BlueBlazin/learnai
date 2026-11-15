import asyncio
from dataclasses import dataclass

RETRYABLE_STATUS = frozenset({429, 503})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_s: float = 0.5
    cap_s: float = 8.0
    deadline_s: float = 30.0
    idempotent: bool = True


def retry_after_s(headers: dict[str, str]) -> float | None:
    """The delay a Retry-After header asks for, or None if absent or not delay-seconds."""
    for name, value in headers.items():
        if name.lower() == "retry-after":  # header names are case-insensitive
            text = value.strip()
            if text.isascii() and text.isdigit():
                return float(int(text))
            return None  # an HTTP-date or garbage: fall back to backoff
    return None


async def call_with_retry(send, policy: RetryPolicy, uniforms: list[float], budget=None):
    loop = asyncio.get_running_loop()
    started = loop.time()
    retries = 0  # retries made so far; the next backoff ceiling is base * 2**retries
    drawn = 0  # uniforms consumed so far
    while True:
        response, error = None, None
        try:
            response = await send()
        except ConnectionError as exc:
            error = exc

        if error is not None:
            # The server may have done the work: only an idempotent request is resent.
            overloaded, retryable = True, policy.idempotent
        else:
            overloaded = retryable = response.status in RETRYABLE_STATUS
        if budget is not None:
            if overloaded:
                budget.failure()
            elif response.status < 400:
                budget.success()

        give_up = (
            not retryable
            or retries + 1 >= policy.max_attempts
            or (budget is not None and not budget.allows_retry())
        )
        if not give_up:
            delay = None if response is None else retry_after_s(response.headers)
            if delay is None:
                # Full jitter: uniform between 0 and the capped exponential ceiling.
                delay = uniforms[drawn] * min(policy.cap_s, policy.base_s * 2**retries)
                drawn += 1
            # A retry that would start at or after the deadline is not worth waiting for.
            give_up = (loop.time() - started) + delay >= policy.deadline_s
        if give_up:
            if error is not None:
                raise error
            return response
        await asyncio.sleep(delay)
        retries += 1
