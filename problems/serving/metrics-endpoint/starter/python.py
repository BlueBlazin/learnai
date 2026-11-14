import json
import time

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
    """The state behind GET /metrics. One instance per app; design its fields yourself.

    It has to hold: the number of running requests, the prompt-token and
    generation-token totals, the number of POST /generate responses per status in
    STATUSES, and the time-to-first-token histogram over TTFT_BUCKETS (a count per
    bucket, the sum of all observations, and their number).
    """

    def __init__(self):
        raise NotImplementedError

    def observe_ttft(self, seconds: float) -> None:
        """Record one time to first token in the histogram.

        An observation belongs to every bucket whose upper bound is greater than or
        equal to it (`le` is "less than or equal"), and always to +Inf.
        """
        raise NotImplementedError

    def render(self) -> str:
        """The metrics in the Prometheus text exposition format.

        One group of lines per metric: `# TYPE <name> <counter|gauge|histogram>`
        first (a `# HELP <name> <text>` line is optional), then the samples,
        `<name> <value>` or `<name>{<label>="<value>"} <value>`. Histogram samples are
        `<name>_bucket{le="<bound>"}` in increasing order of bound and ending with
        le="+Inf", each holding the number of observations <= its bound, then
        `<name>_sum` and `<name>_count`. Every line, the last included, ends with "\\n".
        """
        raise NotImplementedError


def parse_request(body: bytes) -> tuple[str, int]:
    """Validate the body of POST /generate and return (prompt, max_tokens).

    The body must be a JSON object with a string "prompt" and, optionally, an integer
    "max_tokens" >= 1 (default DEFAULT_MAX_TOKENS; booleans are not integers). Raise
    BadRequest otherwise.
    """
    raise NotImplementedError


def make_app(lm, clock=time.monotonic):
    """Build the ASGI app with POST /generate and GET /metrics.

    lm: the model.
        `lm.stream(prompt, max_tokens)` returns an async iterator of the generated
        pieces (str), one generated token each; iterating it may raise RuntimeError
        at any point, which means the generation failed. `await stream.aclose()` stops it.
        `lm.count_tokens(text)` is the number of tokens in a prompt.
    clock: a function returning the current time in seconds. Read time only through
        it; the tests pass their own clock.

    POST /generate:
        invalid body -> 400 {"error": "<message>"}; the model is not called.
        the model fails -> 500 {"error": "<message>"}.
        otherwise -> 200 {"text": <the pieces joined>, "prompt_tokens": <int>,
                          "completion_tokens": <the number of pieces>}.
    GET /metrics: 200, content type CONTENT_TYPE, body `metrics.render()`.
    """
    metrics = Metrics()

    async def generate(request):
        """`await request.body()` is the request body (bytes). Return a JSONResponse."""
        raise NotImplementedError

    async def scrape(request):
        return Response(metrics.render(), media_type=CONTENT_TYPE)

    return Starlette(
        routes=[
            Route("/generate", generate, methods=["POST"]),
            Route("/metrics", scrape, methods=["GET"]),
        ]
    )
