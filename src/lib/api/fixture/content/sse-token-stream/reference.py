import json
from collections.abc import AsyncIterator, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def create_app(generate: Callable[[str, int], AsyncIterator[str]]) -> Starlette:
    """Return an ASGI app. `generate(prompt, max_tokens)` yields token strings."""

    async def stream(request: Request):
        raise NotImplementedError

    return Starlette(routes=[Route("/v1/stream", stream, methods=["POST"])])
