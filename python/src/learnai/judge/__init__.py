"""The test authoring API. See docs/PROBLEM_AUTHORING.md.

from learnai.judge import case, check

@case("matches the reference on a random batch", public=True)
def random_batch(ctx):
    ctx.check("softmax", ctx.randn(4, 8), axis=-1)
"""

from learnai.judge import check
from learnai.judge.context import Context
from learnai.judge.errors import CheckFailure
from learnai.judge.registry import Case, case
from learnai.judge.service import SSEEvent, iter_sse, parse_sse, read_sse, serve
from learnai.judge.simclock import SimClock, Trace, TraceEvent, percentile

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
