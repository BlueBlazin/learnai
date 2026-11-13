import json

from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route


async def stream(request):
    words = request.query_params.get("prompt", "").split()

    async def events():
        for word in words:
            yield f"data: {json.dumps({'token': word})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


app = Starlette(routes=[Route("/stream", stream)])
