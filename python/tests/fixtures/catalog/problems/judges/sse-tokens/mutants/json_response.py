# BUG: returns all tokens in one JSON body instead of a server-sent event stream.
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def stream(request):
    return JSONResponse({"tokens": request.query_params.get("prompt", "").split()})


app = Starlette(routes=[Route("/stream", stream)])
