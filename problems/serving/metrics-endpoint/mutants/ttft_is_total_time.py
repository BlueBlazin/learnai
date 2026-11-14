# BUG: the histogram observes the time until the generation has finished instead of the time until the first piece arrived.
import json
import time
from bisect import bisect_left
from itertools import accumulate

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

DEFAULT_MAX_TOKENS = 16
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5)  # upper bounds in seconds; +Inf is implied
STATUSES = ("200", "400", "500")
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class BadRequest(Exception):
    """The request body is not acceptable; the message says why."""


class Metrics:
    def __init__(self):
        self.running = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        # Every status is present from the start, at 0: a series that appears only
        # with its first event cannot be told apart from a broken exporter.
        self.requests = dict.fromkeys(STATUSES, 0)
        # Observations per bucket (not cumulative); the last entry is the +Inf bucket.
        self.ttft_counts = [0] * (len(TTFT_BUCKETS) + 1)
        self.ttft_sum = 0.0

    def observe_ttft(self, seconds: float) -> None:
        # `le` means less than or equal: a value on a bound belongs to that bucket.
        # bisect_left finds the first bound >= seconds, or len(TTFT_BUCKETS) for +Inf.
        self.ttft_counts[bisect_left(TTFT_BUCKETS, seconds)] += 1
        self.ttft_sum += seconds

    def render(self) -> str:
        """The text exposition: one group per metric, its TYPE line first."""
        ttft = "vllm:time_to_first_token_seconds"
        # Buckets are cumulative on the wire: each counts the observations <= its
        # bound, so the +Inf bucket counts all of them and equals _count.
        cumulative = list(accumulate(self.ttft_counts))
        bounds = [*TTFT_BUCKETS, float("inf")]
        lines = [
            "# HELP vllm:num_requests_running Number of requests currently being generated.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running {self.running}",
            "# HELP vllm:prompt_tokens_total Number of prompt tokens processed.",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total {self.prompt_tokens}",
            "# HELP vllm:generation_tokens_total Number of generation tokens processed.",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total {self.generation_tokens}",
            "# HELP http_requests_total Responses of POST /generate by status code.",
            "# TYPE http_requests_total counter",
            *(f'http_requests_total{{status="{s}"}} {n}' for s, n in self.requests.items()),
            f"# HELP {ttft} Histogram of time to first token in seconds.",
            f"# TYPE {ttft} histogram",
            *(
                f'{ttft}_bucket{{le="{number(bound)}"}} {count}'
                for bound, count in zip(bounds, cumulative, strict=True)
            ),
            f"{ttft}_sum {number(self.ttft_sum)}",
            f"{ttft}_count {cumulative[-1]}",
        ]
        return "\n".join(lines) + "\n"  # the last line must end with a line feed too


def number(value: float) -> str:
    """A float as the exposition format writes it."""
    return "+Inf" if value == float("inf") else repr(float(value))


def parse_request(body: bytes) -> tuple[str, int]:
    """Return (prompt, max_tokens), or raise BadRequest saying what is wrong."""
    try:
        payload = json.loads(body)
    except ValueError:
        raise BadRequest("the body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise BadRequest("'prompt' is required and must be a string")
    max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("'max_tokens' must be an integer >= 1")
    return prompt, max_tokens


def make_app(lm, clock=time.monotonic):
    metrics = Metrics()

    async def generate(request):
        try:
            prompt, max_tokens = parse_request(await request.body())
        except BadRequest as exc:
            metrics.requests["400"] += 1
            return JSONResponse({"error": str(exc)}, status_code=400)

        started = clock()
        prompt_tokens = lm.count_tokens(prompt)
        metrics.prompt_tokens += prompt_tokens
        metrics.running += 1
        pieces = []
        stream = lm.stream(prompt, max_tokens)
        try:
            async for piece in stream:
                pieces.append(piece)
                metrics.generation_tokens += 1
        except RuntimeError as exc:  # the model failed: a 500, and the metrics still add up
            metrics.requests["500"] += 1
            return JSONResponse({"error": f"generation failed: {exc}"}, status_code=500)
        finally:
            # On every way out, or one failed request shows as "running" forever.
            metrics.running -= 1
            await stream.aclose()
        metrics.requests["200"] += 1
        if pieces:
            metrics.observe_ttft(clock() - started)
        return JSONResponse(
            {
                "text": "".join(pieces),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(pieces),
            }
        )

    async def scrape(request):
        return Response(metrics.render(), media_type=CONTENT_TYPE)

    return Starlette(
        routes=[
            Route("/generate", generate, methods=["POST"]),
            Route("/metrics", scrape, methods=["GET"]),
        ]
    )
