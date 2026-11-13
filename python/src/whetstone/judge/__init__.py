"""The test authoring API. See docs/PROBLEM_AUTHORING.md.

from whetstone.judge import case, check

@case("matches the reference on a random batch", public=True)
def random_batch(ctx):
    ctx.check("softmax", ctx.randn(4, 8), axis=-1)
"""

from whetstone.judge import check
from whetstone.judge.context import Context
from whetstone.judge.errors import CheckFailure
from whetstone.judge.registry import Case, case
from whetstone.judge.service import SSEEvent, iter_sse, parse_sse, read_sse, serve
from whetstone.judge.simclock import SimClock, Trace, TraceEvent, percentile

__all__ = [
    "Case",
    "CheckFailure",
    "Context",
    "SSEEvent",
    "SimClock",
    "Trace",
    "TraceEvent",
    "case",
    "check",
    "iter_sse",
    "parse_sse",
    "percentile",
    "read_sse",
    "serve",
]
