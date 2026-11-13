# BUG: builds the endpoint with Starlette (routing, StreamingResponse) instead of the raw ASGI interface, which the statement rules out.
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_MAX_TOKENS = 16


def make_app(token_source):
    async def generate(request):
        try:
            payload = json.loads(await request.body())
        except ValueError:
            return JSONResponse({"error": "the body is not valid JSON"}, status_code=400)
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            return JSONResponse({"error": "'prompt' must be a string"}, status_code=400)
        max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            return JSONResponse({"error": "'max_tokens' must be an integer >= 1"}, status_code=400)

        async def events():
            index = 0
            async for token in token_source(payload["prompt"], max_tokens):
                yield f"data: {json.dumps({'token': token, 'index': index})}\n\n"
                index += 1
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/generate", generate, methods=["POST"])])
